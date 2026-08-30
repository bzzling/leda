import leda;
import spar;
import std;

namespace {

constexpr std::array<char, 8> magic{'L', 'E', 'D', 'A', 'W', 'E', 'B', '\0'};
constexpr std::uint32_t format_version{1};
constexpr std::uint32_t header_bytes{192};
constexpr std::uint32_t dtype_float32{1};
constexpr std::uint64_t payload_alignment{64};
constexpr std::string_view expected_checkpoint_sha{
    "a94e2a15977c811b2bc8a7bb1e87be45510e3d350f08b21ef599644ba3fd1bed"};

class Sha256 final {
public:
  void update(std::span<const std::byte> bytes) {
    bit_count_ += static_cast<std::uint64_t>(bytes.size()) * 8U;
    for (const std::byte value : bytes) {
      block_[block_size_++] = static_cast<std::uint8_t>(value);
      if (block_size_ == block_.size()) {
        transform();
        block_size_ = 0;
      }
    }
  }

  [[nodiscard]] std::array<std::uint8_t, 32> finish() {
    const std::uint64_t original_bits{bit_count_};
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0U);
      transform();
      block_size_ = 0;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0U);
    for (std::size_t index{}; index < 8; ++index) {
      block_[63 - index] = static_cast<std::uint8_t>(original_bits >> (index * 8U));
    }
    transform();
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word{}; word < state_.size(); ++word) {
      for (std::size_t byte{}; byte < 4; ++byte) {
        digest[word * 4 + byte] = static_cast<std::uint8_t>(state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

private:
  static constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
      0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
      0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
      0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
      0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
      0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
      0xc67178f2U};

  static std::uint32_t rotate(std::uint32_t value, unsigned amount) {
    return std::rotr(value, static_cast<int>(amount));
  }

  void transform() {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index{}; index < 16; ++index) {
      words[index] = static_cast<std::uint32_t>(block_[index * 4]) << 24U |
                     static_cast<std::uint32_t>(block_[index * 4 + 1]) << 16U |
                     static_cast<std::uint32_t>(block_[index * 4 + 2]) << 8U |
                     static_cast<std::uint32_t>(block_[index * 4 + 3]);
    }
    for (std::size_t index{16}; index < words.size(); ++index) {
      const std::uint32_t s0{rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^
                             (words[index - 15] >> 3U)};
      const std::uint32_t s1{rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^
                             (words[index - 2] >> 10U)};
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (std::size_t index{}; index < words.size(); ++index) {
      const std::uint32_t upper{rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)};
      const std::uint32_t choice{(e & f) ^ (~e & g)};
      const std::uint32_t first{h + upper + choice + constants[index] + words[index]};
      const std::uint32_t lower{rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)};
      const std::uint32_t majority{(a & b) ^ (a & c) ^ (b & c)};
      const std::uint32_t second{lower + majority};
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{};
  std::uint64_t bit_count_{};
};

std::string hexadecimal(std::span<const std::uint8_t> bytes) {
  constexpr char digits[]{"0123456789abcdef"};
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const std::uint8_t value : bytes) {
    result += digits[value >> 4U];
    result += digits[value & 0x0fU];
  }
  return result;
}

std::array<std::uint8_t, 32> file_sha256(const std::filesystem::path& path) {
  std::ifstream stream{path, std::ios::binary};
  if (!stream) {
    throw std::runtime_error{"Unable to open file for SHA-256: " + path.string()};
  }
  Sha256 sha;
  std::array<char, 1U << 20U> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count{stream.gcount()};
    if (count > 0) {
      sha.update(std::as_bytes(std::span{buffer}.first(static_cast<std::size_t>(count))));
    }
  }
  if (!stream.eof()) {
    throw std::runtime_error{"Failed while hashing file: " + path.string()};
  }
  return sha.finish();
}

class Writer final {
public:
  explicit Writer(const std::filesystem::path& path)
      : stream_{path, std::ios::binary | std::ios::trunc} {
    if (!stream_) {
      throw std::runtime_error{"Unable to create LEDAWEB temporary artifact"};
    }
  }

  void bytes(std::span<const std::byte> value) {
    stream_.write(reinterpret_cast<const char*>(value.data()),
                  static_cast<std::streamsize>(value.size()));
    if (!stream_) {
      throw std::runtime_error{"LEDAWEB artifact write failed"};
    }
    position_ += value.size();
  }
  void bytes(std::span<const char> value) {
    bytes(std::as_bytes(value));
  }
  void u32(std::uint32_t value) {
    std::array<std::byte, 4> encoded{};
    for (std::size_t index{}; index < encoded.size(); ++index) {
      encoded[index] = static_cast<std::byte>(value >> (index * 8U));
    }
    bytes(encoded);
  }
  void u64(std::uint64_t value) {
    std::array<std::byte, 8> encoded{};
    for (std::size_t index{}; index < encoded.size(); ++index) {
      encoded[index] = static_cast<std::byte>(value >> (index * 8U));
    }
    bytes(encoded);
  }
  void f32(float value) {
    u32(std::bit_cast<std::uint32_t>(value));
  }
  void zero(std::uint64_t count) {
    constexpr std::array<std::byte, 4096> zeros{};
    while (count != 0) {
      const std::size_t chunk{
          static_cast<std::size_t>(std::min<std::uint64_t>(count, zeros.size()))};
      bytes(std::span{zeros}.first(chunk));
      count -= chunk;
    }
  }
  void pad_to(std::uint64_t target) {
    if (target < position_) {
      throw std::logic_error{"LEDAWEB writer passed an earlier target offset"};
    }
    zero(target - position_);
  }
  [[nodiscard]] std::uint64_t position() const noexcept {
    return position_;
  }
  void finish() {
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error{"LEDAWEB artifact flush failed"};
    }
    stream_.close();
    if (!stream_) {
      throw std::runtime_error{"LEDAWEB artifact close failed"};
    }
  }

private:
  std::ofstream stream_;
  std::uint64_t position_{};
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0 || value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
    throw std::overflow_error{"LEDAWEB alignment overflow"};
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

struct TensorRecord final {
  std::string name;
  std::vector<std::uint32_t> shape;
  spar::Tensor tensor;
  std::uint64_t payload_offset{};
  std::uint64_t metadata_bytes{};
};

std::vector<TensorRecord> records(leda::Leda& model) {
  std::vector<TensorRecord> result;
  std::set<std::string> names;
  std::uint64_t elements{};
  for (auto& named : leda::named_parameters(model)) {
    const spar::Tensor host{named.parameter.tensor().to(spar::Device::cpu()).contiguous()};
    if (host.dtype() != spar::DType::Float32) {
      throw std::invalid_argument{"LEDAWEB v1 exports only Float32 parameters"};
    }
    if (!names.emplace(named.name).second || named.name.empty() || named.name.size() > 4096) {
      throw std::runtime_error{"Invalid or duplicate Leda parameter name"};
    }
    std::vector<std::uint32_t> shape;
    for (const auto dimension : host.shape().dimensions()) {
      if (dimension <= 0 ||
          static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error{"LEDAWEB tensor dimension is out of range"};
      }
      shape.push_back(static_cast<std::uint32_t>(dimension));
    }
    if (host.numel() > std::numeric_limits<std::uint64_t>::max() - elements) {
      throw std::overflow_error{"LEDAWEB parameter count overflow"};
    }
    elements += host.numel();
    const std::uint64_t unpadded{32U + shape.size() * 4U + named.name.size()};
    result.push_back({named.name, std::move(shape), host, 0, align_up(unpadded, 8)});
  }
  if (elements != 40'385'024U || result.size() != 134U) {
    throw std::runtime_error{"LEDAWEB export parameter inventory does not match Leda Demo v0"};
  }
  return result;
}

void write_float_payload(Writer& writer, const spar::Tensor& tensor) {
  const auto values{tensor.span<float>()};
  if constexpr (std::endian::native == std::endian::little) {
    writer.bytes(std::as_bytes(values));
  } else {
    for (const float value : values) {
      writer.f32(value);
    }
  }
}

void export_artifact(const std::filesystem::path& checkpoint, const std::filesystem::path& output) {
  const auto checkpoint_digest{file_sha256(checkpoint)};
  if (hexadecimal(checkpoint_digest) != expected_checkpoint_sha) {
    throw std::invalid_argument{"Source checkpoint SHA-256 is not the frozen Leda Demo v0"};
  }
  auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
  leda::Leda model{leda::Leda::from_decoder(leda::leda_demo_v0(), std::move(loaded.model))};
  auto tensors{records(model)};
  const std::uint64_t metadata_offset{header_bytes};
  const std::uint64_t metadata_bytes{std::accumulate(
      tensors.begin(), tensors.end(), std::uint64_t{},
      [](std::uint64_t sum, const TensorRecord& record) { return sum + record.metadata_bytes; })};
  const std::uint64_t payload_offset{align_up(metadata_offset + metadata_bytes, payload_alignment)};
  std::uint64_t payload_bytes{};
  for (auto& record : tensors) {
    record.payload_offset = payload_offset + payload_bytes;
    payload_bytes += record.tensor.nbytes();
  }

  std::filesystem::path temporary{output};
  temporary += ".tmp";
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
  try {
    Writer writer{temporary};
    writer.bytes(magic);
    writer.u32(format_version);
    writer.u32(header_bytes);
    writer.u32(0); // flags
    writer.u32(dtype_float32);
    const auto config{leda::leda_demo_v0()};
    writer.u32(static_cast<std::uint32_t>(config.vocab_size));
    writer.u32(static_cast<std::uint32_t>(config.model_dim));
    writer.u32(static_cast<std::uint32_t>(config.hidden_dim));
    writer.u32(static_cast<std::uint32_t>(config.num_layers));
    writer.u32(static_cast<std::uint32_t>(config.num_query_heads));
    writer.u32(static_cast<std::uint32_t>(config.num_kv_heads));
    writer.u32(512);
    writer.u32(static_cast<std::uint32_t>(tensors.size()));
    writer.f32(static_cast<float>(config.norm_epsilon));
    writer.f32(static_cast<float>(config.qk_norm_epsilon));
    writer.f32(static_cast<float>(config.rope_theta));
    writer.u32(0);
    writer.u64(metadata_offset);
    writer.u64(metadata_bytes);
    writer.u64(payload_offset);
    writer.u64(payload_bytes);
    writer.bytes(std::as_bytes(std::span{checkpoint_digest}));
    writer.pad_to(header_bytes);

    for (const auto& record : tensors) {
      const std::uint64_t start{writer.position()};
      writer.u32(static_cast<std::uint32_t>(record.name.size()));
      writer.u32(static_cast<std::uint32_t>(record.shape.size()));
      writer.u32(dtype_float32);
      writer.u32(0);
      writer.u64(record.tensor.numel());
      writer.u64(record.payload_offset);
      for (const std::uint32_t dimension : record.shape) {
        writer.u32(dimension);
      }
      writer.bytes(std::span{record.name});
      writer.pad_to(start + record.metadata_bytes);
    }
    writer.pad_to(payload_offset);
    for (const auto& record : tensors) {
      if (writer.position() != record.payload_offset) {
        throw std::logic_error{"LEDAWEB payload offset mismatch"};
      }
      write_float_payload(writer, record.tensor);
    }
    writer.finish();
    std::filesystem::rename(temporary, output);
  } catch (...) {
    std::filesystem::remove(temporary, ignored);
    throw;
  }
  const auto artifact_digest{file_sha256(output)};
  std::println("format=LEDAWEB version=1 tensors={} parameters={} bytes={} sha256={} "
               "source_checkpoint_sha256={}",
               tensors.size(), 40'385'024U, std::filesystem::file_size(output),
               hexadecimal(artifact_digest), hexadecimal(checkpoint_digest));
}

int run(int argc, char** argv) {
  if (argc != 3) {
    throw std::invalid_argument{"usage: leda_export_web CHECKPOINT OUTPUT.ledaweb"};
  }
  export_artifact(argv[1], argv[2]);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_export_web: " << error.what() << '\n';
    return 1;
  }
}
