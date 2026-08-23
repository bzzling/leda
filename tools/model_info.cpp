import leda;
import spar;
import std;

namespace {

struct Contributions final {
  std::uint64_t embedding;
  std::uint64_t attention;
  std::uint64_t qk_norm;
  std::uint64_t block_norms;
  std::uint64_t swiglu;
  std::uint64_t final_norm;
};

std::uint64_t multiply(std::uint64_t left, std::uint64_t right) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error{"Analytical Parameter count overflow"};
  }
  return left * right;
}

std::uint64_t add(std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error{"Analytical Parameter count overflow"};
  }
  return left + right;
}

Contributions analytical(const leda::LedaConfig& config) {
  const std::uint64_t d{config.model_dim};
  const std::uint64_t hidden{config.hidden_dim};
  const std::uint64_t layers{config.num_layers};
  const std::uint64_t head_dim{config.model_dim / config.num_query_heads};
  const std::uint64_t kv_dim{multiply(config.num_kv_heads, head_dim)};
  return {.embedding = multiply(config.vocab_size, d),
          .attention =
              multiply(layers, add(multiply(2, multiply(d, d)), multiply(2, multiply(d, kv_dim)))),
          .qk_norm = config.qk_norm ? multiply(layers, multiply(2, head_dim)) : 0,
          .block_norms = multiply(layers, multiply(2, d)),
          .swiglu = multiply(layers, multiply(3, multiply(d, hidden))),
          .final_norm = d};
}

std::uint64_t total(const Contributions& values) {
  return add(add(add(values.embedding, values.attention), add(values.qk_norm, values.block_norms)),
             add(values.swiglu, values.final_norm));
}

leda::LedaConfig config(std::string_view name, std::size_t vocab_size) {
  if (name == "A") {
    return leda::leda_tiny(vocab_size);
  }
  if (name == "B") {
    return leda::leda_small(vocab_size);
  }
  if (name == "C") {
    return {.vocab_size = vocab_size,
            .model_dim = 384,
            .hidden_dim = 1152,
            .num_layers = 12,
            .num_query_heads = 6,
            .num_kv_heads = 2};
  }
  if (name == "D") {
    return {.vocab_size = vocab_size,
            .model_dim = 512,
            .hidden_dim = 1536,
            .num_layers = 12,
            .num_query_heads = 8,
            .num_kv_heads = 2};
  }
  throw std::invalid_argument{"model must be A, B, C, or D"};
}

std::size_t parse_vocab(std::string_view text) {
  std::size_t value{};
  const auto parsed{std::from_chars(text.data(), text.data() + text.size(), value)};
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) {
    throw std::invalid_argument{"vocabulary size must be positive"};
  }
  return value;
}

int run(int argc, char** argv) {
  if (argc != 2) {
    throw std::invalid_argument{"usage: leda_model_info MODEL_VOCAB_SIZE"};
  }
  const std::size_t vocab_size{parse_vocab(argv[1])};
  for (const std::string_view name : {"A", "B", "C", "D"}) {
    spar::Random random{2901};
    const leda::LedaConfig model_config{config(name, vocab_size)};
    static_cast<void>(leda::decoder_config(model_config));
    leda::Leda model{model_config, random};
    const leda::ModelStatistics statistics{leda::model_statistics(model)};
    const leda::AdamWMemoryEstimate memory{leda::adamw_memory_estimate(model)};
    const Contributions formula{analytical(model_config)};
    const std::uint64_t analytical_total{total(formula)};
    const std::int64_t mismatch{static_cast<std::int64_t>(statistics.total_parameters) -
                                static_cast<std::int64_t>(analytical_total)};
    std::println("model={} d_model={} hidden={} layers={} hq={} hkv={} parameters={} analytical={} "
                 "mismatch={} parameter_bytes={} gradient_bytes={} first_moment_bytes={} "
                 "second_moment_bytes={} persistent_bytes={} embedding={} attention={} qk_norm={} "
                 "block_norms={} swiglu={} final_norm={}",
                 name, model_config.model_dim, model_config.hidden_dim, model_config.num_layers,
                 model_config.num_query_heads, model_config.num_kv_heads,
                 statistics.total_parameters, analytical_total, mismatch, memory.parameter_bytes,
                 memory.gradient_bytes, memory.first_moment_bytes, memory.second_moment_bytes,
                 memory.persistent_training_bytes, formula.embedding, formula.attention,
                 formula.qk_norm, formula.block_norms, formula.swiglu, formula.final_norm);
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_model_info: " << error.what() << '\n';
    return 1;
  }
}
