module leda.inference;

import std;
import leda.model;
import spar.dtype;
import spar.nn.decoder;
import spar.random;
import spar.tensor;
import spar.tokenizer.byte_bpe;

using namespace std;

namespace leda {
namespace {

size_t validated_capacity(size_t capacity) {
  if (capacity == 0 || capacity > leda_v0_max_context) {
    throw invalid_argument{"Leda v0 inference capacity must be in [1,512]"};
  }
  return capacity;
}

vector<double> final_logits(const spar::Tensor& logits) {
  if (logits.rank() != 3 || logits.shape()[0] != 1 || logits.shape()[1] <= 0 ||
      logits.shape()[2] <= 0) {
    throw invalid_argument{"Sampling requires non-empty batch-1 logits with shape [1,T,V]"};
  }
  if (logits.dtype() != spar::DType::Float32 && logits.dtype() != spar::DType::Float64) {
    throw invalid_argument{"Sampling requires floating-point logits"};
  }
  const spar::Tensor host{logits.to(spar::Device::cpu()).detach().contiguous()};
  const size_t length{static_cast<size_t>(host.shape()[1])};
  const size_t vocabulary{static_cast<size_t>(host.shape()[2])};
  vector<double> result(vocabulary);
  if (host.dtype() == spar::DType::Float32) {
    const auto values{host.span<float>()};
    transform(values.end() - static_cast<ptrdiff_t>(vocabulary), values.end(), result.begin(),
              [](float value) { return static_cast<double>(value); });
  } else {
    const auto values{host.span<double>()};
    copy(values.begin() + static_cast<ptrdiff_t>((length - 1) * vocabulary), values.end(),
         result.begin());
  }
  if (ranges::any_of(result, [](double value) { return !isfinite(value); })) {
    throw runtime_error{"Sampling received non-finite logits"};
  }
  return result;
}

vector<size_t> descending_indices(const vector<double>& values) {
  vector<size_t> indices(values.size());
  iota(indices.begin(), indices.end(), 0);
  ranges::sort(indices, [&values](size_t left, size_t right) {
    return values[left] > values[right] || (values[left] == values[right] && left < right);
  });
  return indices;
}

vector<double> probabilities(vector<double> logits, double temperature, optional<size_t> top_k,
                             optional<double> top_p) {
  if (!isfinite(temperature) || temperature <= 0.0) {
    throw invalid_argument{"Sampling temperature must be finite and positive"};
  }
  if (top_k && (*top_k == 0 || *top_k > logits.size())) {
    throw invalid_argument{"Sampling top-k must be in [1,vocabulary_size]"};
  }
  if (top_p && (!isfinite(*top_p) || *top_p <= 0.0 || *top_p > 1.0)) {
    throw invalid_argument{"Sampling top-p must be finite and in (0,1]"};
  }
  for (double& value : logits) {
    value /= temperature;
  }
  vector<size_t> order{descending_indices(logits)};
  const size_t retained{top_k.value_or(logits.size())};
  const double maximum{logits[order.front()]};
  vector<double> result(logits.size(), 0.0);
  double total{};
  for (size_t rank{0}; rank < retained; ++rank) {
    const size_t token{order[rank]};
    result[token] = exp(logits[token] - maximum);
    total += result[token];
  }
  if (!isfinite(total) || total <= 0.0) {
    throw runtime_error{"Sampling softmax normalization failed"};
  }
  for (double& value : result) {
    value /= total;
  }

  if (top_p && *top_p < 1.0) {
    double cumulative{};
    size_t nucleus_size{};
    for (; nucleus_size < retained; ++nucleus_size) {
      cumulative += result[order[nucleus_size]];
      if (cumulative >= *top_p) {
        ++nucleus_size;
        break;
      }
    }
    for (size_t rank{nucleus_size}; rank < retained; ++rank) {
      result[order[rank]] = 0.0;
    }
    for (double& value : result) {
      value /= cumulative;
    }
  }
  return result;
}

} // namespace

LedaInferenceSession::LedaInferenceSession(const Leda& model, size_t batch_size, size_t capacity)
    : session_{model.decoder(), batch_size, validated_capacity(capacity)} {}

spar::Tensor LedaInferenceSession::prefill(const spar::Tensor& token_ids) {
  return session_.prefill(token_ids);
}

spar::Tensor LedaInferenceSession::decode(const spar::Tensor& token_ids) {
  return session_.decode(token_ids);
}

size_t LedaInferenceSession::current_length() const noexcept {
  return session_.current_length();
}
size_t LedaInferenceSession::capacity() const noexcept {
  return session_.capacity();
}
spar::Device LedaInferenceSession::device() const noexcept {
  return session_.device();
}
const spar::nn::DecoderKVCache& LedaInferenceSession::cache() const noexcept {
  return session_.cache();
}

Sampler::Sampler(SamplingConfig config) : config_{std::move(config)}, random_{config_.seed} {
  if (!config_.greedy) {
    if (!isfinite(config_.temperature) || config_.temperature <= 0.0) {
      throw invalid_argument{"Sampling temperature must be finite and positive"};
    }
    if (config_.top_k && *config_.top_k == 0) {
      throw invalid_argument{"Sampling top-k must be positive when enabled"};
    }
    if (config_.top_p &&
        (!isfinite(*config_.top_p) || *config_.top_p <= 0.0 || *config_.top_p > 1.0)) {
      throw invalid_argument{"Sampling top-p must be finite and in (0,1]"};
    }
  }
}

spar::tokenizer::TokenId Sampler::sample(const spar::Tensor& logits) {
  vector<double> values{final_logits(logits)};
  if (values.size() > static_cast<size_t>(numeric_limits<spar::tokenizer::TokenId>::max())) {
    throw overflow_error{"Logit vocabulary exceeds TokenId"};
  }
  if (config_.greedy) {
    const vector<size_t> order{descending_indices(values)};
    return static_cast<spar::tokenizer::TokenId>(order.front());
  }
  const vector<double> distribution{
      probabilities(std::move(values), config_.temperature, config_.top_k, config_.top_p)};
  const double draw{random_.uniform_double()};
  double cumulative{};
  size_t fallback{};
  for (size_t token{0}; token < distribution.size(); ++token) {
    if (distribution[token] > 0.0) {
      fallback = token;
    }
    cumulative += distribution[token];
    if (draw < cumulative) {
      return static_cast<spar::tokenizer::TokenId>(token);
    }
  }
  return static_cast<spar::tokenizer::TokenId>(fallback);
}

const SamplingConfig& Sampler::config() const noexcept {
  return config_;
}

vector<TokenProbability> top_candidates(const spar::Tensor& logits, size_t count) {
  vector<double> values{final_logits(logits)};
  if (count == 0 || count > values.size()) {
    throw invalid_argument{"Top-candidate count must be in [1,vocabulary_size]"};
  }
  const vector<double> distribution{probabilities(values, 1.0, nullopt, nullopt)};
  const vector<size_t> order{descending_indices(values)};
  vector<TokenProbability> result;
  result.reserve(count);
  for (size_t rank{0}; rank < count; ++rank) {
    result.push_back(
        {static_cast<spar::tokenizer::TokenId>(order[rank]), distribution[order[rank]]});
  }
  return result;
}

} // namespace leda
