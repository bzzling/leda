import spar;
import std;

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void usage() {
  throw std::invalid_argument{"usage: leda_train_tokenizer STRATIFIED_TRAIN_SAMPLE_PATHS "
                              "VALIDATION_PATHS OUTPUT [TARGET_VOCAB=8192] "
                              "[MAX_TRAINING_BYTES=134217728]"};
}

std::uint64_t parse_positive(std::string_view text, std::string_view name) {
  std::uint64_t value{};
  const auto result{std::from_chars(text.data(), text.data() + text.size(), value)};
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0) {
    throw std::invalid_argument{std::string{name} + " must be a positive uint64"};
  }
  return value;
}

std::vector<std::filesystem::path> read_paths(const std::filesystem::path& manifest) {
  std::ifstream stream{manifest};
  if (!stream) {
    throw std::runtime_error{"Unable to open path manifest: " + manifest.string()};
  }
  std::vector<std::filesystem::path> result;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    std::filesystem::path path{line};
    if (path.is_relative()) {
      path = manifest.parent_path() / path;
    }
    result.push_back(std::move(path));
  }
  if (!stream.eof() || result.empty()) {
    throw std::runtime_error{"Path manifest is unreadable or empty: " + manifest.string()};
  }
  return result;
}

std::string read_document(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size{std::filesystem::file_size(path, error)};
  if (error || size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
      size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error{"Document is too large: " + path.string()};
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  std::ifstream stream{path, std::ios::binary};
  stream.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (stream.gcount() != static_cast<std::streamsize>(result.size())) {
    throw std::runtime_error{"Unable to read complete document: " + path.string()};
  }
  return result;
}

std::optional<std::uint64_t> peak_rss_kib() {
  std::ifstream stream{"/proc/self/status"};
  std::string key;
  while (stream >> key) {
    if (key == "VmHWM:") {
      std::uint64_t value{};
      std::string unit;
      if (stream >> value >> unit && unit == "kB") {
        return value;
      }
      return std::nullopt;
    }
    stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return std::nullopt;
}

std::pair<std::uint64_t, std::uint64_t> measure(const spar::tokenizer::ByteBPETokenizer& tokenizer,
                                                std::span<const std::filesystem::path> paths) {
  std::uint64_t bytes{};
  std::uint64_t tokens{};
  for (const auto& path : paths) {
    const std::string document{read_document(path)};
    const auto encoded{tokenizer.encode(document)};
    if (document.size() > std::numeric_limits<std::uint64_t>::max() - bytes ||
        encoded.size() > std::numeric_limits<std::uint64_t>::max() - tokens) {
      throw std::overflow_error{"Tokenizer evaluation count overflow"};
    }
    bytes += static_cast<std::uint64_t>(document.size());
    tokens += static_cast<std::uint64_t>(encoded.size());
  }
  return {bytes, tokens};
}

void inspect(const spar::tokenizer::ByteBPETokenizer& tokenizer, std::string_view text) {
  const auto tokens{tokenizer.encode(text)};
  std::print("inspect {:?}: {} tokens [", text, tokens.size());
  for (std::size_t index{}; index < tokens.size(); ++index) {
    std::print("{}{}", index == 0 ? "" : ",", tokens[index]);
  }
  std::println("]");
}

int run(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    usage();
  }
  const std::filesystem::path train_manifest{argv[1]};
  const std::filesystem::path validation_manifest{argv[2]};
  const std::filesystem::path output{argv[3]};
  const std::uint64_t target_vocab{argc >= 5 ? parse_positive(argv[4], "TARGET_VOCAB") : 8192};
  const std::uint64_t maximum_bytes{argc >= 6 ? parse_positive(argv[5], "MAX_TRAINING_BYTES")
                                              : 134'217'728};
  if (target_vocab > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{"TARGET_VOCAB exceeds size_t"};
  }
  const auto train_paths{read_paths(train_manifest)};
  const auto validation_paths{read_paths(validation_manifest)};
  std::vector<std::string> sample;
  std::uint64_t sample_bytes{};
  for (const auto& path : train_paths) {
    std::string document{read_document(path)};
    if (!sample.empty() &&
        (sample_bytes >= maximum_bytes || document.size() > maximum_bytes - sample_bytes)) {
      break;
    }
    sample_bytes += static_cast<std::uint64_t>(document.size());
    sample.push_back(std::move(document));
    if (sample_bytes >= maximum_bytes) {
      break;
    }
  }
  const auto started{Clock::now()};
  const auto tokenizer{spar::tokenizer::train_byte_bpe(
      sample,
      {.target_vocab_size = static_cast<std::size_t>(target_vocab), .min_pair_frequency = 2})};
  const double seconds{std::chrono::duration<double>(Clock::now() - started).count()};
  spar::tokenizer::save_tokenizer(output, tokenizer);
  std::uint64_t sample_tokens{};
  for (const auto& document : sample) {
    sample_tokens += static_cast<std::uint64_t>(tokenizer.encode(document).size());
  }
  const auto [validation_bytes, validation_tokens]{measure(tokenizer, validation_paths)};
  if (sample_bytes == 0 || sample_tokens == 0 || validation_bytes == 0 || validation_tokens == 0) {
    throw std::runtime_error{"Tokenizer evaluation corpus must contain nonempty encoded text"};
  }
  const auto rss{peak_rss_kib()};
  std::println("training_documents={} training_bytes={} target_vocab={} "
               "actual_vocab={} wall_s={:.3f} peak_rss_kib={}",
               sample.size(), sample_bytes, target_vocab, tokenizer.vocab_size(), seconds,
               rss ? std::to_string(*rss) : "unavailable");
  std::println("train_bytes_per_token={:.6f} validation_bytes_per_token={:.6f} "
               "train_compression_vs_bytes={:.6f} "
               "validation_compression_vs_bytes={:.6f}",
               static_cast<double>(sample_bytes) / static_cast<double>(sample_tokens),
               static_cast<double>(validation_bytes) / static_cast<double>(validation_tokens),
               static_cast<double>(sample_tokens) / static_cast<double>(sample_bytes),
               static_cast<double>(validation_tokens) / static_cast<double>(validation_bytes));
  for (const std::string_view text :
       {"interleukin-6", "CD19", "β-catenin", "N-acetylcysteine", "TP53", "phosphorylation",
        "double-stranded DNA", "mg/kg", "p < 0.05"}) {
    inspect(tokenizer, text);
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_train_tokenizer: " << error.what() << '\n';
    return 1;
  }
}
