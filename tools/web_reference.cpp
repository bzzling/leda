import leda;
import spar;
import std;

namespace {

constexpr std::array<char, 8> magic{'L', 'E', 'D', 'A', 'R', 'E', 'F', '\0'};
constexpr std::uint32_t version{1};
constexpr std::uint32_t header_bytes{128};
constexpr std::uint32_t vocabulary{8193};
constexpr std::array<std::byte, 32> checkpoint_sha{
    std::byte{0xa9}, std::byte{0x4e}, std::byte{0x2a}, std::byte{0x15}, std::byte{0x97},
    std::byte{0x7c}, std::byte{0x81}, std::byte{0x1b}, std::byte{0x2b}, std::byte{0xc8},
    std::byte{0xa7}, std::byte{0xbb}, std::byte{0x1e}, std::byte{0x87}, std::byte{0xbe},
    std::byte{0x45}, std::byte{0x51}, std::byte{0x0e}, std::byte{0x3d}, std::byte{0x35},
    std::byte{0x0f}, std::byte{0x08}, std::byte{0xb2}, std::byte{0x1e}, std::byte{0xf5},
    std::byte{0x99}, std::byte{0x64}, std::byte{0x4b}, std::byte{0xa3}, std::byte{0xfd},
    std::byte{0x1b}, std::byte{0xed}};
constexpr std::array<std::byte, 32> tokenizer_sha{
    std::byte{0x04}, std::byte{0x34}, std::byte{0x56}, std::byte{0x87}, std::byte{0x33},
    std::byte{0x8f}, std::byte{0x13}, std::byte{0xb1}, std::byte{0xb8}, std::byte{0xe8},
    std::byte{0xd2}, std::byte{0x19}, std::byte{0xda}, std::byte{0xe2}, std::byte{0x65},
    std::byte{0x09}, std::byte{0x4a}, std::byte{0x70}, std::byte{0x8b}, std::byte{0xf8},
    std::byte{0xd1}, std::byte{0xb4}, std::byte{0x09}, std::byte{0x54}, std::byte{0x17},
    std::byte{0x48}, std::byte{0xcb}, std::byte{0x2d}, std::byte{0x89}, std::byte{0x42},
    std::byte{0x49}, std::byte{0x70}};

class Writer final {
public:
  explicit Writer(const std::filesystem::path& path) : stream_{path, std::ios::binary} {
    if (!stream_) {
      throw std::runtime_error{"Unable to create LEDAREF fixture"};
    }
  }

  void bytes(std::span<const std::byte> values) {
    stream_.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size()));
    if (!stream_) {
      throw std::runtime_error{"LEDAREF write failed"};
    }
    position_ += values.size();
  }
  void bytes(std::string_view values) {
    bytes(std::as_bytes(std::span{values}));
  }
  void u32(std::uint32_t value) {
    std::array<std::byte, 4> encoded{};
    for (std::size_t index{}; index < encoded.size(); ++index) {
      encoded[index] = static_cast<std::byte>(value >> (index * 8U));
    }
    bytes(encoded);
  }
  void f32(float value) {
    u32(std::bit_cast<std::uint32_t>(value));
  }
  void pad(std::size_t alignment) {
    while (position_ % alignment != 0) {
      bytes(std::array<std::byte, 1>{});
    }
  }
  void zero_to(std::size_t target) {
    if (target < position_) {
      throw std::logic_error{"LEDAREF header overflow"};
    }
    while (position_ < target) {
      bytes(std::array<std::byte, 1>{});
    }
  }
  void finish() {
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error{"LEDAREF flush failed"};
    }
  }

private:
  std::ofstream stream_;
  std::size_t position_{};
};

spar::Tensor token_tensor(std::span<const spar::tokenizer::TokenId> tokens, spar::Device device) {
  spar::Tensor host{spar::Shape{1, static_cast<std::int64_t>(tokens.size())}, spar::DType::Int64};
  std::ranges::transform(tokens, host.span<std::int64_t>().begin(),
                         [](auto token) { return static_cast<std::int64_t>(token); });
  return host.to(device);
}

std::vector<float> final_logits(const spar::Tensor& logits) {
  if (logits.rank() != 3 || logits.shape()[0] != 1 || logits.shape()[2] != vocabulary) {
    throw std::runtime_error{"Unexpected reference logit shape"};
  }
  const spar::Tensor host{logits.to(spar::Device::cpu()).detach().contiguous()};
  if (host.dtype() != spar::DType::Float32) {
    throw std::runtime_error{"LEDAREF v1 requires Float32 logits"};
  }
  const auto values{host.span<float>()};
  return {values.end() - static_cast<std::ptrdiff_t>(vocabulary), values.end()};
}

std::uint32_t greedy(std::span<const float> logits) {
  std::size_t best{};
  for (std::size_t token{1}; token < logits.size(); ++token) {
    if (logits[token] > logits[best]) {
      best = token;
    }
  }
  return static_cast<std::uint32_t>(best);
}

struct TokenizerCase final {
  std::string bytes;
  std::vector<spar::tokenizer::TokenId> tokens;
};

std::vector<TokenizerCase> tokenizer_cases(const spar::tokenizer::ByteBPETokenizer& tokenizer) {
  std::vector<std::string> inputs{
      "Activation of naive CD8+ T cells requires",
      "Activation of naïve CD8+ T cells requires",
      "β-oxidation of α-keto acids at 1.23×10⁻⁴ mol L⁻¹",
      "H₂O + CO₂ ⇌ H₂CO₃; μ = 0.15 M",
      "Ångström-scale coupling: E = mc²",
      "6.022e23 particles mol^-1",
      std::string{"raw\0bytes\xFF\x80", 11},
  };
  std::vector<TokenizerCase> result;
  for (auto& input : inputs) {
    result.push_back({input, tokenizer.encode(input)});
  }
  return result;
}

std::vector<spar::tokenizer::TokenId> prompt(std::size_t case_index, std::size_t length) {
  std::vector<spar::tokenizer::TokenId> result(length);
  for (std::size_t position{}; position < length; ++position) {
    result[position] = static_cast<spar::tokenizer::TokenId>(
        (case_index * 977U + position * 37U + position * position * 3U + 11U) % 8192U);
  }
  return result;
}

void write_stage(Writer& writer, std::uint32_t input_token, const spar::Tensor& output) {
  const std::vector<float> logits{final_logits(output)};
  writer.u32(input_token);
  writer.u32(greedy(logits));
  for (const float value : logits) {
    if (!std::isfinite(value)) {
      throw std::runtime_error{"Native reference produced a non-finite logit"};
    }
    writer.f32(value);
  }
}

void generate(const std::filesystem::path& checkpoint, const std::filesystem::path& tokenizer_path,
              const std::filesystem::path& output, spar::Device device) {
  const auto tokenizer{spar::tokenizer::load_tokenizer(tokenizer_path)};
  if (tokenizer.vocab_size() != 8192U) {
    throw std::invalid_argument{"LEDAREF requires the frozen 8192-token tokenizer"};
  }
  auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
  leda::Leda model{leda::Leda::from_decoder(leda::leda_demo_v0(), std::move(loaded.model))};
  auto parameters{leda::parameters(model)};
  spar::nn::move_to(std::span<spar::nn::Parameter>{parameters}, device);
  spar::set_matmul_precision(device, spar::MatmulPrecision::Full);

  const auto text_cases{tokenizer_cases(tokenizer)};
  constexpr std::array<std::size_t, 8> lengths{1, 17, 32, 127, 128, 256, 400, 500};
  std::filesystem::path temporary{output};
  temporary += ".tmp";
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
  try {
    Writer writer{temporary};
    writer.bytes(std::as_bytes(std::span{magic}));
    writer.u32(version);
    writer.u32(header_bytes);
    writer.u32(static_cast<std::uint32_t>(text_cases.size()));
    writer.u32(static_cast<std::uint32_t>(lengths.size()));
    writer.u32(vocabulary);
    writer.u32(0);
    writer.bytes(checkpoint_sha);
    writer.bytes(tokenizer_sha);
    writer.zero_to(header_bytes);

    for (const auto& item : text_cases) {
      writer.u32(static_cast<std::uint32_t>(item.bytes.size()));
      writer.u32(static_cast<std::uint32_t>(item.tokens.size()));
      writer.bytes(item.bytes);
      for (const auto token : item.tokens) {
        writer.u32(token);
      }
      writer.pad(8);
    }

    for (std::size_t case_index{}; case_index < lengths.size(); ++case_index) {
      const auto tokens{prompt(case_index, lengths[case_index])};
      const std::size_t decode_count{lengths[case_index] == 500 ? 12U : 3U};
      writer.u32(static_cast<std::uint32_t>(tokens.size()));
      writer.u32(static_cast<std::uint32_t>(decode_count));
      for (const auto token : tokens) {
        writer.u32(token);
      }
      leda::LedaInferenceSession session{model};
      write_stage(writer, std::numeric_limits<std::uint32_t>::max(),
                  session.prefill(token_tensor(tokens, device)));
      for (std::size_t step{}; step < decode_count; ++step) {
        const auto token{
            static_cast<spar::tokenizer::TokenId>((case_index * 313U + step * 997U + 17U) % 8192U)};
        write_stage(writer, token, session.decode(token_tensor(std::span{&token, 1}, device)));
      }
      std::println("reference_geometry={}+{} complete", tokens.size(), decode_count);
    }
    writer.finish();
    std::filesystem::rename(temporary, output);
  } catch (...) {
    std::filesystem::remove(temporary, ignored);
    throw;
  }
  std::println("format=LEDAREF version=1 tokenizer_cases={} model_cases={} bytes={}",
               text_cases.size(), lengths.size(), std::filesystem::file_size(output));
}

int run(int argc, char** argv) {
  if (argc != 5) {
    throw std::invalid_argument{
        "usage: leda_web_reference CHECKPOINT TOKENIZER OUTPUT.ledaref cpu|cuda"};
  }
  const std::string_view selected{argv[4]};
  const spar::Device device{selected == "cpu" ? spar::Device::cpu()
                            : selected == "cuda"
                                ? spar::Device::cuda(0)
                                : throw std::invalid_argument{"device must be cpu or cuda"}};
  generate(argv[1], argv[2], argv[3], device);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_web_reference: " << error.what() << '\n';
    return 1;
  }
}
