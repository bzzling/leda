#if defined(LEDA_BENCHMARK_CUDA)
#include <cuda_runtime_api.h>
#endif

import leda;
import spar;
import std;

namespace {

using Clock = std::chrono::steady_clock;

struct Measurement final {
  double prefill_ms{};
  double first_token_ms{};
  double cached_decode_tokens_per_second{};
  double cached_completion_ms{};
  double uncached_tokens_per_second{};
  double uncached_completion_ms{};
  double speedup{};
  std::uint64_t peak_vram_bytes{};
};

std::uint64_t free_vram(const spar::Device& device) {
#if defined(LEDA_BENCHMARK_CUDA)
  if (device.is_cuda()) {
    std::size_t free{};
    std::size_t total{};
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
      throw std::runtime_error{"cudaMemGetInfo failed"};
    }
    return static_cast<std::uint64_t>(free);
  }
#else
  static_cast<void>(device);
#endif
  return 0;
}

spar::Tensor tensor(std::span<const spar::tokenizer::TokenId> tokens, spar::Device device) {
  spar::Tensor host{spar::Shape{1, static_cast<std::int64_t>(tokens.size())}, spar::DType::Int64};
  std::ranges::transform(tokens, host.span<std::int64_t>().begin(),
                         [](auto token) { return static_cast<std::int64_t>(token); });
  return host.to(device);
}

leda::Leda load_model(const std::filesystem::path& checkpoint) {
  auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
  return leda::Leda::from_decoder(leda::leda_demo_v0(), std::move(loaded.model));
}

template <typename Function> double milliseconds(Function&& function, const spar::Device& device) {
  const auto start{Clock::now()};
  std::invoke(std::forward<Function>(function));
  spar::synchronize(device);
  return std::chrono::duration<double, std::milli>{Clock::now() - start}.count();
}

Measurement benchmark(const std::filesystem::path& checkpoint, spar::Device device,
                      spar::MatmulPrecision precision, std::size_t prompt_length,
                      std::size_t generated_tokens, bool measure_memory) {
  if (prompt_length == 0 || generated_tokens == 0 ||
      prompt_length + generated_tokens > leda::leda_v0_max_context) {
    throw std::invalid_argument{"Benchmark requires prompt>0, generation>0, and total<=512"};
  }
  spar::synchronize(device);
  const std::uint64_t initial_free{free_vram(device)};
  std::uint64_t minimum_free{initial_free};
  const auto sample_vram = [&] {
    if (device.is_cuda()) {
      minimum_free = std::min(minimum_free, free_vram(device));
    }
  };

  leda::Leda model{load_model(checkpoint)};
  auto parameters{leda::parameters(model)};
  spar::nn::move_to(std::span<spar::nn::Parameter>{parameters}, device);
  spar::set_matmul_precision(device, precision);
  sample_vram();

  std::vector<spar::tokenizer::TokenId> prompt(prompt_length);
  for (std::size_t index{0}; index < prompt.size(); ++index) {
    prompt[index] = static_cast<spar::tokenizer::TokenId>((index * 37U + 11U) % 8192U);
  }
  leda::Sampler greedy{{.greedy = true}};

  {
    spar::InferenceMode guard;
    const spar::Tensor warmup{model.forward(tensor(std::span{prompt.data(), 1}, device))};
    static_cast<void>(greedy.sample(warmup));
  }
  spar::synchronize(device);
  sample_vram();

  if (measure_memory) {
    std::uint64_t inference_bytes{};
    std::uint64_t ordinary_bytes{};
    const std::uint64_t model_free{free_vram(device)};
    {
      spar::InferenceMode guard;
      const spar::Tensor output{model.forward(tensor(prompt, device))};
      static_cast<void>(greedy.sample(output));
      spar::synchronize(device);
      if (device.is_cuda()) {
        inference_bytes = model_free - free_vram(device);
      }
    }
    spar::synchronize(device);
    {
      const spar::Tensor output{model.forward(tensor(prompt, device))};
      static_cast<void>(greedy.sample(output));
      spar::synchronize(device);
      if (device.is_cuda()) {
        ordinary_bytes = model_free - free_vram(device);
      }
      if (!output.requires_grad()) {
        throw std::logic_error{"Ordinary benchmark forward did not retain autograd"};
      }
    }
    std::println(
        "memory prompt={} inference_forward_bytes={} ordinary_forward_bytes={} saved_bytes={}",
        prompt_length, inference_bytes, ordinary_bytes,
        ordinary_bytes >= inference_bytes ? ordinary_bytes - inference_bytes : 0);
    sample_vram();
  }

  Measurement result;
  leda::LedaInferenceSession cached{model};
  spar::Tensor cached_logits{spar::Shape{1, 1, 8193}, spar::DType::Float32, device};
  const auto cached_start{Clock::now()};
  result.prefill_ms =
      milliseconds([&] { cached_logits = cached.prefill(tensor(prompt, device)); }, device);
  const auto first_start{Clock::now()};
  auto token{greedy.sample(cached_logits)};
  spar::synchronize(device);
  result.first_token_ms =
      result.prefill_ms +
      std::chrono::duration<double, std::milli>{Clock::now() - first_start}.count();
  sample_vram();

  const auto decode_start{Clock::now()};
  for (std::size_t index{1}; index < generated_tokens; ++index) {
    cached_logits = cached.decode(tensor(std::span{&token, 1}, device));
    token = greedy.sample(cached_logits);
    sample_vram();
  }
  spar::synchronize(device);
  const double decode_seconds{std::chrono::duration<double>{Clock::now() - decode_start}.count()};
  result.cached_decode_tokens_per_second =
      generated_tokens > 1 ? static_cast<double>(generated_tokens - 1) / decode_seconds : 0.0;
  result.cached_completion_ms =
      std::chrono::duration<double, std::milli>{Clock::now() - cached_start}.count();

  std::vector<spar::tokenizer::TokenId> growing{prompt};
  const auto uncached_start{Clock::now()};
  for (std::size_t index{0}; index < generated_tokens; ++index) {
    spar::Tensor full_logits{spar::Shape{1, 1, 8193}, spar::DType::Float32, device};
    {
      spar::InferenceMode guard;
      full_logits = model.forward(tensor(growing, device));
    }
    token = greedy.sample(full_logits);
    if (index + 1 < generated_tokens) {
      growing.push_back(token);
    }
    sample_vram();
  }
  spar::synchronize(device);
  const double uncached_seconds{
      std::chrono::duration<double>{Clock::now() - uncached_start}.count()};
  result.uncached_tokens_per_second = static_cast<double>(generated_tokens) / uncached_seconds;
  result.uncached_completion_ms = uncached_seconds * 1000.0;
  const double cached_tokens_per_second{static_cast<double>(generated_tokens) /
                                        (result.cached_completion_ms / 1000.0)};
  result.speedup = cached_tokens_per_second / result.uncached_tokens_per_second;
  result.peak_vram_bytes = device.is_cuda() ? initial_free - minimum_free : 0;
  return result;
}

int run(int argc, char** argv) {
  if (argc < 6 || argc > 7) {
    throw std::invalid_argument{
        "usage: leda_inference_benchmark CHECKPOINT cpu|cuda full|fp16 PROMPT_LENGTH "
        "GENERATED_TOKENS [--memory]"};
  }
  const spar::Device device{std::string_view{argv[2]} == "cpu" ? spar::Device::cpu()
                            : std::string_view{argv[2]} == "cuda"
                                ? spar::Device::cuda(0)
                                : throw std::invalid_argument{"device must be cpu or cuda"}};
  const spar::MatmulPrecision precision{
      std::string_view{argv[3]} == "full" ? spar::MatmulPrecision::Full
      : std::string_view{argv[3]} == "fp16"
          ? spar::MatmulPrecision::Float16Inputs
          : throw std::invalid_argument{"precision must be full or fp16"}};
  const auto parse_size = [](std::string_view text) {
    std::size_t value{};
    const auto parsed{std::from_chars(text.data(), text.data() + text.size(), value)};
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
      throw std::invalid_argument{"invalid benchmark size"};
    }
    return value;
  };
  const Measurement result{benchmark(argv[1], device, precision, parse_size(argv[4]),
                                     parse_size(argv[5]),
                                     argc == 7 && std::string_view{argv[6]} == "--memory")};
  std::println("prompt={} generated={} device={} precision={} prefill_ms={:.6f} "
               "first_token_ms={:.6f} cached_decode_tok_s={:.6f} cached_completion_ms={:.6f} "
               "uncached_tok_s={:.6f} uncached_completion_ms={:.6f} speedup={:.6f} "
               "peak_vram_bytes={}",
               argv[4], argv[5], argv[2], argv[3], result.prefill_ms, result.first_token_ms,
               result.cached_decode_tokens_per_second, result.cached_completion_ms,
               result.uncached_tokens_per_second, result.uncached_completion_ms, result.speedup,
               result.peak_vram_bytes);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_inference_benchmark: " << error.what() << '\n';
    return 1;
  }
}
