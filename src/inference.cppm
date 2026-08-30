export module leda.inference;

import std;
export import leda.model;
export import spar.nn.decoder;
export import spar.tokenizer.byte_bpe;

export namespace leda {

inline constexpr std::size_t leda_v0_max_context{512};

/// Move-only inference state for one Leda completion session.
class LedaInferenceSession final {
public:
  explicit LedaInferenceSession(const Leda& model, std::size_t batch_size = 1,
                                std::size_t capacity = leda_v0_max_context);

  LedaInferenceSession(const LedaInferenceSession&) = delete;
  LedaInferenceSession& operator=(const LedaInferenceSession&) = delete;
  LedaInferenceSession(LedaInferenceSession&&) noexcept = default;
  LedaInferenceSession& operator=(LedaInferenceSession&&) noexcept = default;

  [[nodiscard]] spar::Tensor prefill(const spar::Tensor& token_ids);
  [[nodiscard]] spar::Tensor decode(const spar::Tensor& token_ids);

  [[nodiscard]] std::size_t current_length() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] spar::Device device() const noexcept;
  [[nodiscard]] const spar::nn::DecoderKVCache& cache() const noexcept;

private:
  spar::nn::DecoderInferenceSession session_;
};

struct SamplingConfig final {
  bool greedy{false};
  double temperature{1.0};
  std::optional<std::size_t> top_k{};
  std::optional<double> top_p{};
  std::uint64_t seed{0};
};

struct TokenProbability final {
  spar::tokenizer::TokenId token;
  double probability;
};

/// Deterministic CPU sampler for the final [B,T,V] logit row. Phase 32 supports batch 1.
class Sampler final {
public:
  explicit Sampler(SamplingConfig config);

  [[nodiscard]] spar::tokenizer::TokenId sample(const spar::Tensor& logits);
  [[nodiscard]] const SamplingConfig& config() const noexcept;

private:
  SamplingConfig config_;
  spar::Random random_;
};

/// Returns the highest-probability entries from the unfiltered model distribution.
[[nodiscard]] std::vector<TokenProbability> top_candidates(const spar::Tensor& logits,
                                                           std::size_t count);

} // namespace leda
