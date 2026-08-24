import leda;
import spar;
import spar.loss.cross_entropy;
import std;

namespace {

struct Document final {
  std::string id;
  std::string domain;
  std::string source;
  std::filesystem::path path;
};

std::uint64_t parse_u64(std::string_view text, std::string_view name) {
  std::uint64_t value{};
  const auto parsed{std::from_chars(text.data(), text.data() + text.size(), value)};
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) {
    throw std::invalid_argument{std::string{name} + " must be a positive uint64"};
  }
  return value;
}

std::map<std::string, std::string> fields(const std::filesystem::path& path) {
  std::ifstream stream{path};
  std::map<std::string, std::string> result;
  std::string line;
  while (std::getline(stream, line)) {
    const std::size_t equals{line.find('=')};
    if (!line.empty() &&
        (equals == std::string::npos || equals == 0 || equals + 1 == line.size() ||
         !result.emplace(line.substr(0, equals), line.substr(equals + 1)).second)) {
      throw std::invalid_argument{"Invalid or duplicate run-spec field"};
    }
  }
  return result;
}

const std::string& required(const std::map<std::string, std::string>& values,
                            std::string_view key) {
  const auto found{values.find(std::string{key})};
  if (found == values.end()) {
    throw std::invalid_argument{"Missing run-spec field: " + std::string{key}};
  }
  return found->second;
}

std::vector<Document> documents(const std::filesystem::path& path) {
  std::ifstream stream{path};
  if (!stream) {
    throw std::runtime_error{"Unable to open validation document TSV: " + path.string()};
  }
  std::string line;
  if (!std::getline(stream, line) ||
      line != "canonical_document_id\tbroad_domain\tsource_family\tlocal_document_path") {
    throw std::invalid_argument{"Invalid validation document TSV header"};
  }
  std::vector<Document> result;
  while (std::getline(stream, line)) {
    const std::size_t first{line.find('\t')};
    const std::size_t second{first == std::string::npos ? first : line.find('\t', first + 1)};
    const std::size_t third{second == std::string::npos ? second : line.find('\t', second + 1)};
    if (first == std::string::npos || second == std::string::npos || third == std::string::npos ||
        line.find('\t', third + 1) != std::string::npos) {
      throw std::invalid_argument{"Invalid validation document TSV row"};
    }
    result.push_back({line.substr(0, first), line.substr(first + 1, second - first - 1),
                      line.substr(second + 1, third - second - 1), line.substr(third + 1)});
  }
  if (result.empty()) {
    throw std::invalid_argument{"Validation document TSV is empty"};
  }
  return result;
}

std::string read_document(const std::filesystem::path& path) {
  std::ifstream stream{path, std::ios::binary};
  if (!stream) {
    throw std::runtime_error{"Unable to open validation document: " + path.string()};
  }
  return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

std::string csv(std::string_view value) {
  std::string result{"\""};
  for (const char character : value) {
    result += character;
    if (character == '"') {
      result += '"';
    }
  }
  return result + '"';
}

struct Loss final {
  double sum{};
  std::uint64_t targets{};
};

Loss evaluate_batch(const leda::Leda& model,
                    std::span<const std::vector<spar::tokenizer::TokenId>> sequences,
                    const spar::Device& device) {
  const std::size_t length{sequences.front().size()};
  if (sequences.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      length > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error{"Document evaluation batch shape exceeds int64"};
  }
  spar::Tensor host{
      spar::Shape{static_cast<std::int64_t>(sequences.size()), static_cast<std::int64_t>(length)},
      spar::DType::Int64};
  auto values{host.span<std::int64_t>()};
  for (std::size_t row{}; row < sequences.size(); ++row) {
    if (sequences[row].size() != length) {
      throw std::logic_error{"Document evaluation batch has inconsistent lengths"};
    }
    for (std::size_t column{}; column < length; ++column) {
      values[row * length + column] = sequences[row][column];
    }
  }
  const spar::Tensor input{host.to(device)};
  const spar::Tensor losses{spar::loss::language_model_cross_entropy(model.forward(input), input,
                                                                     spar::loss::Reduction::None)};
  const spar::Tensor result{losses.to(spar::Device::cpu())};
  return {.sum = std::accumulate(result.span<float>().begin(), result.span<float>().end(), 0.0),
          .targets = static_cast<std::uint64_t>(result.numel())};
}

int run(int argc, char** argv) {
  if (argc != 8) {
    throw std::invalid_argument{"usage: leda_evaluate_documents RUN_SPEC CHECKPOINT TOKENIZER "
                                "VALIDATION_DOCUMENTS_TSV OUTPUT_CSV CONTEXT_LENGTH BATCH_SIZE"};
  }
  const auto spec{fields(argv[1])};
  const std::size_t vocab{
      static_cast<std::size_t>(parse_u64(required(spec, "model_vocab_size"), "model_vocab_size"))};
  const leda::LedaConfig config{
      .vocab_size = vocab,
      .model_dim = static_cast<std::size_t>(parse_u64(required(spec, "model_dim"), "model_dim")),
      .hidden_dim = static_cast<std::size_t>(parse_u64(required(spec, "hidden_dim"), "hidden_dim")),
      .num_layers = static_cast<std::size_t>(parse_u64(required(spec, "num_layers"), "num_layers")),
      .num_query_heads =
          static_cast<std::size_t>(parse_u64(required(spec, "num_query_heads"), "num_query_heads")),
      .num_kv_heads =
          static_cast<std::size_t>(parse_u64(required(spec, "num_kv_heads"), "num_kv_heads"))};
  const std::size_t context{static_cast<std::size_t>(parse_u64(argv[6], "CONTEXT_LENGTH"))};
  const std::size_t batch_size{static_cast<std::size_t>(parse_u64(argv[7], "BATCH_SIZE"))};
  if (context < 2) {
    throw std::invalid_argument{"CONTEXT_LENGTH must be at least two"};
  }
  const std::string& precision{required(spec, "precision")};
  if (precision != "full" && precision != "fp16") {
    throw std::invalid_argument{"run-spec precision must be full or fp16"};
  }
  auto loaded{spar::checkpoint::load_training_checkpoint(argv[2])};
  const spar::checkpoint::TrainingProgress checkpoint_progress{loaded.progress};
  leda::Leda model{leda::Leda::from_decoder(config, std::move(loaded.model))};
  const spar::Device device{spar::Device::cuda(0)};
  loaded.optimizer.move_to(device);
  spar::set_matmul_precision(device, precision == "fp16" ? spar::MatmulPrecision::Float16Inputs
                                                         : spar::MatmulPrecision::Full);
  const auto tokenizer{spar::tokenizer::load_tokenizer(argv[3])};
  if (tokenizer.vocab_size() + 1 != vocab) {
    throw std::invalid_argument{"Tokenizer/model vocabulary mismatch"};
  }
  std::ofstream output{argv[5]};
  if (!output) {
    throw std::runtime_error{"Unable to create document evaluation CSV"};
  }
  output << "canonical_document_id,broad_domain,source_family,targets,mean_loss\n";
  Loss global;
  for (const Document& document : documents(argv[4])) {
    auto tokens{tokenizer.encode(read_document(document.path))};
    tokens.push_back(static_cast<spar::tokenizer::TokenId>(tokenizer.vocab_size()));
    Loss loss;
    std::vector<std::vector<spar::tokenizer::TokenId>> batch;
    for (std::size_t offset{}; offset + 1 < tokens.size();) {
      const std::size_t length{std::min(context, tokens.size() - offset)};
      std::vector<spar::tokenizer::TokenId> sequence(
          tokens.begin() + static_cast<std::ptrdiff_t>(offset),
          tokens.begin() + static_cast<std::ptrdiff_t>(offset + length));
      if (!batch.empty() && batch.front().size() != sequence.size()) {
        const Loss part{evaluate_batch(model, batch, device)};
        loss.sum += part.sum;
        loss.targets += part.targets;
        batch.clear();
      }
      batch.push_back(std::move(sequence));
      if (batch.size() == batch_size) {
        const Loss part{evaluate_batch(model, batch, device)};
        loss.sum += part.sum;
        loss.targets += part.targets;
        batch.clear();
      }
      offset += length;
    }
    if (!batch.empty()) {
      const Loss part{evaluate_batch(model, batch, device)};
      loss.sum += part.sum;
      loss.targets += part.targets;
    }
    if (loss.targets == 0) {
      throw std::runtime_error{"Validation document produced no objective tokens"};
    }
    global.sum += loss.sum;
    global.targets += loss.targets;
    output << csv(document.id) << ',' << csv(document.domain) << ',' << csv(document.source) << ','
           << loss.targets << ',' << std::setprecision(10)
           << loss.sum / static_cast<double>(loss.targets) << '\n';
  }
  std::println("checkpoint_step={} checkpoint_tokens={} documents={} targets={} "
               "token_weighted_ce={:.8f}",
               checkpoint_progress.global_step, checkpoint_progress.tokens_seen,
               documents(argv[4]).size(), global.targets,
               global.sum / static_cast<double>(global.targets));
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_evaluate_documents: " << error.what() << '\n';
    return 1;
  }
}
