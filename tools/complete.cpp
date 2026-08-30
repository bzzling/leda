import leda;
import spar;
import std;

namespace {

constexpr spar::tokenizer::TokenId eod_token{8192};

struct Options final {
  std::filesystem::path checkpoint;
  std::filesystem::path tokenizer;
  std::string prompt;
  std::size_t max_new_tokens{64};
  bool greedy{false};
  double temperature{1.0};
  std::optional<std::size_t> top_k{};
  std::optional<double> top_p{};
  std::uint64_t seed{0};
  std::size_t show_top{};
  spar::Device device{spar::Device::cpu()};
  spar::MatmulPrecision precision{spar::MatmulPrecision::Full};
};

template <typename T> T number(std::string_view text, std::string_view name) {
  T value{};
  const auto parsed{std::from_chars(text.data(), text.data() + text.size(), value)};
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument{std::string{name} + " has an invalid value"};
  }
  return value;
}

double real_number(std::string_view text, std::string_view name) {
  std::size_t consumed{};
  double value{};
  try {
    value = std::stod(std::string{text}, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument{std::string{name} + " has an invalid value"};
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument{std::string{name} + " has an invalid value"};
  }
  return value;
}

Options options(int argc, char** argv) {
  Options result;
  const auto argument = [&](int& index, std::string_view name) -> std::string_view {
    if (++index >= argc) {
      throw std::invalid_argument{std::string{name} + " requires a value"};
    }
    return argv[index];
  };
  for (int index{1}; index < argc; ++index) {
    const std::string_view flag{argv[index]};
    if (flag == "--checkpoint") {
      result.checkpoint = argument(index, flag);
    } else if (flag == "--tokenizer") {
      result.tokenizer = argument(index, flag);
    } else if (flag == "--prompt") {
      result.prompt = argument(index, flag);
    } else if (flag == "--max-new-tokens") {
      result.max_new_tokens = number<std::size_t>(argument(index, flag), flag);
    } else if (flag == "--temperature") {
      result.temperature = real_number(argument(index, flag), flag);
    } else if (flag == "--top-k") {
      result.top_k = number<std::size_t>(argument(index, flag), flag);
    } else if (flag == "--top-p") {
      result.top_p = real_number(argument(index, flag), flag);
    } else if (flag == "--seed") {
      result.seed = number<std::uint64_t>(argument(index, flag), flag);
    } else if (flag == "--show-top") {
      result.show_top = number<std::size_t>(argument(index, flag), flag);
    } else if (flag == "--device") {
      const std::string_view value{argument(index, flag)};
      if (value == "cpu") {
        result.device = spar::Device::cpu();
      } else if (value == "cuda") {
        result.device = spar::Device::cuda(0);
      } else {
        throw std::invalid_argument{"--device must be cpu or cuda"};
      }
    } else if (flag == "--precision") {
      const std::string_view value{argument(index, flag)};
      if (value == "full") {
        result.precision = spar::MatmulPrecision::Full;
      } else if (value == "fp16") {
        result.precision = spar::MatmulPrecision::Float16Inputs;
      } else {
        throw std::invalid_argument{"--precision must be full or fp16"};
      }
    } else if (flag == "--greedy") {
      result.greedy = true;
    } else {
      throw std::invalid_argument{"Unknown argument: " + std::string{flag}};
    }
  }
  if (result.checkpoint.empty() || result.tokenizer.empty() || result.prompt.empty()) {
    throw std::invalid_argument{
        "usage: leda_complete --checkpoint FILE --tokenizer FILE --prompt TEXT "
        "[--max-new-tokens N] [--greedy | --temperature X --top-k K --top-p P --seed S] "
        "[--show-top N] [--device cpu|cuda] [--precision full|fp16]"};
  }
  if (result.max_new_tokens == 0) {
    throw std::invalid_argument{"--max-new-tokens must be positive"};
  }
  return result;
}

spar::Tensor token_tensor(std::span<const spar::tokenizer::TokenId> tokens, spar::Device device) {
  spar::Tensor host{spar::Shape{1, static_cast<std::int64_t>(tokens.size())}, spar::DType::Int64};
  std::ranges::transform(tokens, host.span<std::int64_t>().begin(),
                         [](auto token) { return static_cast<std::int64_t>(token); });
  return host.to(device);
}

std::string escaped_bytes(std::string_view bytes) {
  constexpr char hexadecimal[]{"0123456789ABCDEF"};
  std::string result;
  for (const char character : bytes) {
    const auto byte{static_cast<unsigned char>(character)};
    switch (byte) {
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    case '\\':
      result += "\\\\";
      break;
    default:
      if (byte >= 0x20U && byte <= 0x7EU) {
        result += static_cast<char>(byte);
      } else {
        result += "\\x";
        result += hexadecimal[byte >> 4U];
        result += hexadecimal[byte & 0x0FU];
      }
    }
  }
  return result;
}

bool continuation(unsigned char byte) {
  return (byte & 0xC0U) == 0x80U;
}

std::size_t utf8_width(unsigned char byte) {
  if (byte < 0x80U) {
    return 1;
  }
  if (byte >= 0xC2U && byte <= 0xDFU) {
    return 2;
  }
  if (byte >= 0xE0U && byte <= 0xEFU) {
    return 3;
  }
  if (byte >= 0xF0U && byte <= 0xF4U) {
    return 4;
  }
  return 0;
}

bool valid_utf8(std::string_view bytes, std::size_t offset, std::size_t width) {
  if (width < 2 || offset + width > bytes.size()) {
    return false;
  }
  const auto first{static_cast<unsigned char>(bytes[offset])};
  for (std::size_t index{1}; index < width; ++index) {
    if (!continuation(static_cast<unsigned char>(bytes[offset + index]))) {
      return false;
    }
  }
  const auto second{static_cast<unsigned char>(bytes[offset + 1])};
  return !(width == 3 &&
           ((first == 0xE0U && second < 0xA0U) || (first == 0xEDU && second >= 0xA0U))) &&
         !(width == 4 &&
           ((first == 0xF0U && second < 0x90U) || (first == 0xF4U && second >= 0x90U)));
}

void emit_stream(std::string_view decoded, std::size_t& consumed, bool final) {
  constexpr char hexadecimal[]{"0123456789ABCDEF"};
  while (consumed < decoded.size()) {
    const auto byte{static_cast<unsigned char>(decoded[consumed])};
    const std::size_t width{utf8_width(byte)};
    if (width == 1) {
      if ((byte < 0x20U && byte != '\n' && byte != '\t') || byte == 0x7FU) {
        std::cout << "\\x" << hexadecimal[byte >> 4U] << hexadecimal[byte & 0x0FU];
      } else {
        std::cout.put(static_cast<char>(byte));
      }
      ++consumed;
    } else if (width != 0 && consumed + width <= decoded.size() &&
               valid_utf8(decoded, consumed, width)) {
      std::cout.write(decoded.data() + static_cast<std::ptrdiff_t>(consumed),
                      static_cast<std::streamsize>(width));
      consumed += width;
    } else if (!final && width != 0 && consumed + width > decoded.size()) {
      break;
    } else {
      std::cout << "\\x" << hexadecimal[byte >> 4U] << hexadecimal[byte & 0x0FU];
      ++consumed;
    }
  }
  std::cout.flush();
}

leda::Leda load_model(const std::filesystem::path& checkpoint) {
  auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
  return leda::Leda::from_decoder(leda::leda_demo_v0(), std::move(loaded.model));
}

int run(int argc, char** argv) {
  const Options selected{options(argc, argv)};
  const auto tokenizer{spar::tokenizer::load_tokenizer(selected.tokenizer)};
  const auto config{leda::leda_demo_v0()};
  if (tokenizer.vocab_size() + 1 != config.vocab_size) {
    throw std::invalid_argument{"Tokenizer/model vocabulary mismatch"};
  }
  const std::vector<spar::tokenizer::TokenId> prompt_tokens{tokenizer.encode(selected.prompt)};
  if (prompt_tokens.empty()) {
    throw std::invalid_argument{"Prompt must encode to at least one token"};
  }
  if (prompt_tokens.size() >= leda::leda_v0_max_context) {
    throw std::length_error{"Prompt leaves no Leda v0 generation capacity"};
  }
  if (selected.max_new_tokens > leda::leda_v0_max_context - prompt_tokens.size()) {
    throw std::length_error{"Prompt plus --max-new-tokens exceeds Leda v0 context 512"};
  }

  leda::Leda model{load_model(selected.checkpoint)};
  auto parameters{leda::parameters(model)};
  spar::nn::move_to(std::span<spar::nn::Parameter>{parameters}, selected.device);
  spar::set_matmul_precision(selected.device, selected.precision);
  leda::LedaInferenceSession session{model};
  spar::Tensor logits{session.prefill(token_tensor(prompt_tokens, selected.device))};

  if (selected.show_top != 0) {
    std::println("Next-token candidates:");
    for (const auto candidate : leda::top_candidates(logits, selected.show_top)) {
      const std::string piece{candidate.token == eod_token ? "<EOD>"
                                                           : escaped_bytes(tokenizer.decode(
                                                                 std::span{&candidate.token, 1}))};
      std::println("  id={} probability={:.8f} bytes=\"{}\"", candidate.token,
                   candidate.probability, piece);
    }
    std::println();
  }

  leda::Sampler sampler{{.greedy = selected.greedy,
                         .temperature = selected.temperature,
                         .top_k = selected.top_k,
                         .top_p = selected.top_p,
                         .seed = selected.seed}};
  std::println("Prompt:\n{}\n\nCompletion:", selected.prompt);
  std::vector<spar::tokenizer::TokenId> generated;
  generated.reserve(selected.max_new_tokens);
  std::size_t emitted_bytes{};
  bool stopped_on_eod{false};
  for (std::size_t step{0}; step < selected.max_new_tokens; ++step) {
    const auto token{sampler.sample(logits)};
    if (token == eod_token) {
      stopped_on_eod = true;
      break;
    }
    generated.push_back(token);
    const std::string decoded{tokenizer.decode(generated)};
    emit_stream(decoded, emitted_bytes, false);
    if (step + 1 < selected.max_new_tokens) {
      logits = session.decode(token_tensor(std::span{&token, 1}, selected.device));
    }
  }
  emit_stream(tokenizer.decode(generated), emitted_bytes, true);
  std::println("\n\nGenerated tokens: {}{}", generated.size(), stopped_on_eod ? " (EOD)" : "");
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_complete: " << error.what() << '\n';
    return 1;
  }
}
