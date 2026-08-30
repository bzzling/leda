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

double continuation_loss(const leda::Leda& model, std::span<const spar::tokenizer::TokenId> tokens,
                         std::size_t prefix_length, const spar::Device& device) {
  if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error{"Prefix diagnostic sequence exceeds int64"};
  }
  spar::Tensor host{spar::Shape{1, static_cast<std::int64_t>(tokens.size())}, spar::DType::Int64};
  auto values{host.span<std::int64_t>()};
  std::ranges::transform(tokens, values.begin(), [](spar::tokenizer::TokenId token) {
    return static_cast<std::int64_t>(token);
  });
  const spar::Tensor input{host.to(device)};
  const spar::Tensor losses{spar::loss::language_model_cross_entropy(model.forward(input), input,
                                                                     spar::loss::Reduction::None)};
  const spar::Tensor host_losses{losses.to(spar::Device::cpu())};
  const auto result{host_losses.span<float>()};
  const std::size_t first{prefix_length - 1};
  const double sum{
      std::accumulate(result.begin() + static_cast<std::ptrdiff_t>(first), result.end(), 0.0)};
  return sum / static_cast<double>(result.size() - first);
}

int run(int argc, char** argv) {
  if (argc != 7) {
    throw std::invalid_argument{"usage: leda_evaluate_prefixes RUN_SPEC CHECKPOINT TOKENIZER "
                                "VALIDATION_DOCUMENTS_TSV OUTPUT_CSV CONTINUATION_TOKENS"};
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
  const std::size_t continuation{
      static_cast<std::size_t>(parse_u64(argv[6], "CONTINUATION_TOKENS"))};
  const std::string& precision{required(spec, "precision")};
  if (precision != "full" && precision != "fp16") {
    throw std::invalid_argument{"run-spec precision must be full or fp16"};
  }
  auto loaded{spar::checkpoint::load_training_checkpoint(argv[2])};
  const spar::checkpoint::TrainingProgress progress{loaded.progress};
  leda::Leda model{leda::Leda::from_decoder(config, std::move(loaded.model))};
  const spar::Device device{spar::Device::cuda(0)};
  loaded.optimizer.move_to(device);
  spar::set_matmul_precision(device, precision == "fp16" ? spar::MatmulPrecision::Float16Inputs
                                                         : spar::MatmulPrecision::Full);
  const auto tokenizer{spar::tokenizer::load_tokenizer(argv[3])};
  if (tokenizer.vocab_size() + 1 != vocab) {
    throw std::invalid_argument{"Tokenizer/model vocabulary mismatch"};
  }
  const std::vector<Document> candidates{documents(argv[4])};
  std::set<std::string> domains;
  for (const Document& document : candidates) {
    domains.insert(document.domain);
  }
  if (domains.size() != 6) {
    throw std::invalid_argument{"Prefix diagnostic requires exactly six validation domains"};
  }
  std::ofstream output{argv[5]};
  if (!output) {
    throw std::runtime_error{"Unable to create prefix diagnostic CSV"};
  }
  output << "checkpoint_step,checkpoint_tokens,prefix_group,canonical_document_id,broad_domain,"
            "source_family,prefix_tokens,continuation_targets,mean_loss\n";
  const std::array<std::size_t, 5> prefix_lengths{32, 64, 128, 256, 400};
  std::size_t examples{};
  for (const std::size_t prefix : prefix_lengths) {
    std::set<std::string> selected;
    for (const std::string& domain : domains) {
      const Document* chosen{};
      std::vector<spar::tokenizer::TokenId> tokens;
      for (const Document& candidate : candidates) {
        if (candidate.domain != domain || selected.contains(candidate.id)) {
          continue;
        }
        auto encoded{tokenizer.encode(read_document(candidate.path))};
        if (encoded.size() >= prefix + continuation) {
          chosen = &candidate;
          tokens = std::move(encoded);
          break;
        }
      }
      if (chosen == nullptr) {
        throw std::runtime_error{"No unused validation document can satisfy prefix group " +
                                 std::to_string(prefix) + " in domain " + domain};
      }
      selected.insert(chosen->id);
      tokens.resize(prefix + continuation);
      const double loss{continuation_loss(model, tokens, prefix, device)};
      if (!std::isfinite(loss)) {
        throw std::runtime_error{"Prefix diagnostic produced non-finite loss"};
      }
      output << progress.global_step << ',' << progress.tokens_seen << ',' << prefix << ','
             << csv(chosen->id) << ',' << csv(chosen->domain) << ',' << csv(chosen->source) << ','
             << prefix << ',' << continuation << ',' << std::setprecision(10) << loss << '\n';
      output.flush();
      ++examples;
      std::println("prefix={} domain={} document={} continuation_ce={:.8f}", prefix, domain,
                   chosen->id, loss);
    }
  }
  std::println("checkpoint_step={} checkpoint_tokens={} examples={} continuation_targets={}",
               progress.global_step, progress.tokens_seen, examples, examples * continuation);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_evaluate_prefixes: " << error.what() << '\n';
    return 1;
  }
}
