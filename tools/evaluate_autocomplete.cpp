import leda;
import spar;
import spar.loss.cross_entropy;
import std;

namespace {

constexpr std::size_t context_capacity{512};
constexpr std::size_t continuation_targets{64};
constexpr std::size_t prefix_examples_per_domain{8};
constexpr std::size_t next_token_chunks_per_domain{20};
constexpr std::uint64_t selection_seed{3301};

struct Metadata final {
  std::string id;
  std::string domain;
  std::string source;
};

struct Document final {
  Metadata metadata;
  std::vector<spar::tokenizer::TokenId> tokens;
};

struct Score final {
  double loss_sum{};
  std::uint64_t targets{};
};

struct NextMetrics final {
  std::uint64_t positions{};
  std::uint64_t top1{};
  std::uint64_t top5{};
  std::uint64_t top10{};
  double correct_probability_sum{};
  double entropy_sum{};
};

struct Chunk final {
  std::size_t document{};
  std::size_t start{};
  std::uint64_t key{};
};

struct PrefixExample final {
  std::size_t document{};
  std::size_t start{};
};

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t begin{};
  while (true) {
    const std::size_t end{line.find('\t', begin)};
    fields.push_back(line.substr(begin, end == std::string_view::npos ? end : end - begin));
    if (end == std::string_view::npos) {
      return fields;
    }
    begin = end + 1;
  }
}

std::vector<Metadata> metadata(const std::filesystem::path& path, std::string_view split) {
  std::ifstream stream{path};
  if (!stream) {
    throw std::runtime_error{"Unable to open mixture metadata: " + path.string()};
  }
  std::string line;
  if (!std::getline(stream, line) ||
      line != "split\tsource_family\tbroad_domain\tlicense\tlocal_document_path") {
    throw std::invalid_argument{"Invalid mixture metadata header"};
  }
  std::vector<Metadata> result;
  while (std::getline(stream, line)) {
    const auto fields{split_tabs(line)};
    if (fields.size() != 5) {
      throw std::invalid_argument{"Invalid mixture metadata row"};
    }
    if (fields[0] == split) {
      const std::filesystem::path document_path{fields[4]};
      result.push_back(
          {document_path.stem().string(), std::string{fields[2]}, std::string{fields[1]}});
    }
  }
  if (result.empty()) {
    throw std::invalid_argument{"Mixture metadata has no rows for requested split"};
  }
  return result;
}

std::vector<Document> documents(const std::filesystem::path& shard, std::vector<Metadata> records) {
  const spar::data::ShardedTokenCorpus corpus{{shard}};
  if (corpus.document_count() != records.size() || corpus.model_vocab_size() != 8193 ||
      corpus.eod_token_id() != 8192) {
    throw std::invalid_argument{"Packed corpus metadata does not match the requested split"};
  }
  if (corpus.token_count() > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{"Packed split does not fit in memory on this platform"};
  }
  std::vector<spar::tokenizer::TokenId> packed(static_cast<std::size_t>(corpus.token_count()));
  corpus.read_tokens(0, packed);
  std::vector<Document> result;
  result.reserve(records.size());
  std::size_t begin{};
  for (std::size_t index{}; index < packed.size(); ++index) {
    if (packed[index] == corpus.eod_token_id()) {
      result.push_back({std::move(records[result.size()]),
                        {packed.begin() + static_cast<std::ptrdiff_t>(begin),
                         packed.begin() + static_cast<std::ptrdiff_t>(index + 1)}});
      begin = index + 1;
    }
  }
  if (begin != packed.size() || result.size() != records.size()) {
    throw std::runtime_error{"Packed split document boundaries are inconsistent"};
  }
  return result;
}

std::uint64_t stable_key(std::string_view id, std::size_t position, std::uint64_t seed) {
  std::uint64_t value{1469598103934665603ULL};
  for (const char character : id) {
    const auto byte{static_cast<unsigned char>(character)};
    value ^= byte;
    value *= 1099511628211ULL;
  }
  value ^= static_cast<std::uint64_t>(position);
  value *= 1099511628211ULL;
  value ^= seed;
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

spar::Tensor token_tensor(std::span<const spar::tokenizer::TokenId> tokens,
                          const spar::Device& device) {
  if (tokens.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error{"Evaluation sequence exceeds the Tensor shape range"};
  }
  spar::Tensor host{spar::Shape{1, static_cast<std::int64_t>(tokens.size())}, spar::DType::Int64};
  std::ranges::transform(tokens, host.span<std::int64_t>().begin(),
                         [](auto token) { return static_cast<std::int64_t>(token); });
  return host.to(device);
}

void accumulate_next(const spar::Tensor& logits, std::span<const spar::tokenizer::TokenId> tokens,
                     std::size_t first_loss, NextMetrics& metrics) {
  const spar::Tensor host{logits.to(spar::Device::cpu()).contiguous()};
  if (host.dtype() != spar::DType::Float32 || host.rank() != 3 || host.shape()[0] != 1 ||
      static_cast<std::size_t>(host.shape()[1]) != tokens.size() || host.shape()[2] != 8193) {
    throw std::logic_error{"Unexpected logits in next-token evaluation"};
  }
  constexpr std::size_t vocabulary{8193};
  const auto values{host.span<float>()};
  for (std::size_t time{first_loss}; time + 1 < tokens.size(); ++time) {
    const float* row{values.data() + time * vocabulary};
    std::array<float, 10> best_values{};
    std::array<std::size_t, 10> best_tokens{};
    best_values.fill(-std::numeric_limits<float>::infinity());
    best_tokens.fill(vocabulary);
    double maximum{-std::numeric_limits<double>::infinity()};
    for (std::size_t token{}; token < vocabulary; ++token) {
      maximum = std::max(maximum, static_cast<double>(row[token]));
      std::size_t rank{};
      while (rank < best_values.size() &&
             (row[token] < best_values[rank] ||
              (row[token] == best_values[rank] && token > best_tokens[rank]))) {
        ++rank;
      }
      if (rank < best_values.size()) {
        for (std::size_t move{best_values.size() - 1}; move > rank; --move) {
          best_values[move] = best_values[move - 1];
          best_tokens[move] = best_tokens[move - 1];
        }
        best_values[rank] = row[token];
        best_tokens[rank] = token;
      }
    }
    double sum{};
    double weighted{};
    for (std::size_t token{}; token < vocabulary; ++token) {
      const double shifted{static_cast<double>(row[token]) - maximum};
      const double weight{std::exp(shifted)};
      sum += weight;
      weighted += weight * shifted;
    }
    if (!std::isfinite(sum) || sum <= 0.0) {
      throw std::runtime_error{"Next-token softmax normalization failed"};
    }
    const std::size_t correct{tokens[time + 1]};
    const auto found{std::ranges::find(best_tokens, correct)};
    const std::size_t rank{static_cast<std::size_t>(found - best_tokens.begin())};
    ++metrics.positions;
    metrics.top1 += rank < 1 ? 1U : 0U;
    metrics.top5 += rank < 5 ? 1U : 0U;
    metrics.top10 += rank < 10 ? 1U : 0U;
    metrics.correct_probability_sum += std::exp(static_cast<double>(row[correct]) - maximum) / sum;
    metrics.entropy_sum += std::log(sum) - weighted / sum;
  }
}

Score score(const leda::Leda& model, std::span<const spar::tokenizer::TokenId> tokens,
            std::size_t target_offset, const spar::Device& device,
            NextMetrics* next_metrics = nullptr) {
  if (tokens.size() < 2 || tokens.size() > context_capacity || target_offset == 0 ||
      target_offset >= tokens.size()) {
    throw std::invalid_argument{"Invalid teacher-forced evaluation geometry"};
  }
  const spar::Tensor input{token_tensor(tokens, device)};
  spar::Tensor logits{spar::Shape{1, 2, 8193}, spar::DType::Float32, device};
  spar::Tensor losses{spar::Shape{1, 1}, spar::DType::Float32, device};
  {
    spar::InferenceMode guard;
    logits = model.forward(input);
    losses = spar::loss::language_model_cross_entropy(logits, input, spar::loss::Reduction::None);
  }
  const spar::Tensor host_losses{losses.to(spar::Device::cpu()).contiguous()};
  const auto values{host_losses.span<float>()};
  const std::size_t first{target_offset - 1};
  const double sum{
      std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(first), values.end(), 0.0)};
  if (next_metrics != nullptr) {
    accumulate_next(logits, tokens, first, *next_metrics);
  }
  return {.loss_sum = sum, .targets = static_cast<std::uint64_t>(values.size() - first)};
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

std::set<std::pair<std::size_t, std::size_t>>
selected_next_chunks(const std::vector<Document>& split) {
  std::map<std::string, std::vector<Chunk>> candidates;
  for (std::size_t document{}; document < split.size(); ++document) {
    const auto& item{split[document]};
    for (std::size_t start{}; start + context_capacity <= item.tokens.size();
         start += context_capacity - 1) {
      candidates[item.metadata.domain].push_back(
          {document, start, stable_key(item.metadata.id, start, selection_seed)});
    }
  }
  std::set<std::pair<std::size_t, std::size_t>> result;
  if (candidates.size() != 6) {
    throw std::runtime_error{"Next-token sample requires exactly six domains"};
  }
  for (auto& [domain, chunks] : candidates) {
    std::ranges::sort(chunks, [](const Chunk& left, const Chunk& right) {
      return std::tie(left.key, left.document, left.start) <
             std::tie(right.key, right.document, right.start);
    });
    if (chunks.size() < next_token_chunks_per_domain) {
      throw std::runtime_error{"Insufficient full chunks for next-token domain " + domain};
    }
    for (const Chunk& chunk : std::span{chunks}.first(next_token_chunks_per_domain)) {
      result.emplace(chunk.document, chunk.start);
    }
  }
  return result;
}

std::vector<PrefixExample> selected_prefix_examples(const std::vector<Document>& split,
                                                    std::string_view domain, std::size_t prefix) {
  struct Candidate final {
    std::uint64_t key;
    std::size_t document;
    std::size_t start;
  };
  std::vector<Candidate> candidates;
  for (std::size_t index{}; index < split.size(); ++index) {
    const Document& item{split[index]};
    if (item.metadata.domain != domain) {
      continue;
    }
    for (std::size_t start{}; start + prefix + continuation_targets < item.tokens.size();
         start += context_capacity) {
      candidates.push_back(
          {stable_key(item.metadata.id, start + prefix * 1'000'003U, selection_seed), index,
           start});
    }
  }
  std::ranges::sort(candidates, [](const Candidate& left, const Candidate& right) {
    return std::tie(left.key, left.document, left.start) <
           std::tie(right.key, right.document, right.start);
  });
  if (candidates.size() < prefix_examples_per_domain) {
    throw std::runtime_error{"Insufficient prefix examples for domain " + std::string{domain}};
  }
  std::vector<PrefixExample> result;
  for (const Candidate& candidate : std::span{candidates}.first(prefix_examples_per_domain)) {
    result.push_back({candidate.document, candidate.start});
  }
  return result;
}

leda::Leda load_model(const std::filesystem::path& checkpoint, const spar::Device& device) {
  auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
  leda::Leda model{leda::Leda::from_decoder(leda::leda_demo_v0(), std::move(loaded.model))};
  auto parameters{leda::parameters(model)};
  spar::nn::move_to(std::span<spar::nn::Parameter>{parameters}, device);
  spar::set_matmul_precision(device, spar::MatmulPrecision::Full);
  return model;
}

int run(int argc, char** argv) {
  if (argc != 7) {
    throw std::invalid_argument{
        "usage: leda_evaluate_autocomplete CHECKPOINT SHARD MIXTURE_TSV validation|test "
        "OUTPUT_DIRECTORY cpu|cuda"};
  }
  const std::string_view split_name{argv[4]};
  if (split_name != "validation" && split_name != "test") {
    throw std::invalid_argument{"Evaluation split must be validation or test"};
  }
  const spar::Device device{std::string_view{argv[6]} == "cpu" ? spar::Device::cpu()
                            : std::string_view{argv[6]} == "cuda"
                                ? spar::Device::cuda(0)
                                : throw std::invalid_argument{"Device must be cpu or cuda"}};
  const std::filesystem::path output_directory{argv[5]};
  std::filesystem::create_directories(output_directory);
  std::vector<Document> split{documents(argv[2], metadata(argv[3], split_name))};
  const auto selected_chunks{selected_next_chunks(split)};
  leda::Leda model{load_model(argv[1], device)};

  std::ofstream document_output{output_directory / "documents.csv"};
  std::ofstream next_output{output_directory / "next-token.csv"};
  std::ofstream prefix_output{output_directory / "prefixes.csv"};
  if (!document_output || !next_output || !prefix_output) {
    throw std::runtime_error{"Unable to create evaluation outputs"};
  }
  document_output
      << "split,document_index,document_id,broad_domain,source_family,targets,loss_sum,mean_loss\n";
  next_output << "split,document_index,document_id,broad_domain,chunk_start,positions,top1,top5,"
                 "top10,correct_probability_sum,entropy_sum\n";
  prefix_output << "split,prefix_tokens,available_context,document_index,document_id,segment_start,"
                   "broad_domain,source_family,continuation_targets,loss_sum,mean_loss\n";

  Score global;
  std::size_t evaluated_chunks{};
  for (std::size_t document_index{}; document_index < split.size(); ++document_index) {
    const Document& document{split[document_index]};
    Score total;
    for (std::size_t start{}; start + 1 < document.tokens.size();) {
      const std::size_t length{std::min(context_capacity, document.tokens.size() - start)};
      NextMetrics next;
      NextMetrics* selected{selected_chunks.contains({document_index, start}) ? &next : nullptr};
      const Score part{
          score(model, std::span{document.tokens}.subspan(start, length), 1, device, selected)};
      total.loss_sum += part.loss_sum;
      total.targets += part.targets;
      if (selected != nullptr) {
        next_output << csv(split_name) << ',' << document_index << ',' << csv(document.metadata.id)
                    << ',' << csv(document.metadata.domain) << ',' << start << ',' << next.positions
                    << ',' << next.top1 << ',' << next.top5 << ',' << next.top10 << ','
                    << std::setprecision(17) << next.correct_probability_sum << ','
                    << next.entropy_sum << '\n';
      }
      start += length - 1;
      ++evaluated_chunks;
    }
    if (total.targets == 0) {
      throw std::runtime_error{"Packed document has no scorable target"};
    }
    global.loss_sum += total.loss_sum;
    global.targets += total.targets;
    document_output << csv(split_name) << ',' << document_index << ',' << csv(document.metadata.id)
                    << ',' << csv(document.metadata.domain) << ',' << csv(document.metadata.source)
                    << ',' << total.targets << ',' << std::setprecision(17) << total.loss_sum << ','
                    << total.loss_sum / static_cast<double>(total.targets) << '\n';
    document_output.flush();
    if ((document_index + 1) % 25 == 0 || document_index + 1 == split.size()) {
      std::println("documents={}/{} targets={} ce={:.8f}", document_index + 1, split.size(),
                   global.targets, global.loss_sum / static_cast<double>(global.targets));
    }
  }

  std::set<std::string> domains;
  for (const Document& document : split) {
    domains.insert(document.metadata.domain);
  }
  constexpr std::array<std::size_t, 5> prefixes{32, 64, 128, 256, 400};
  constexpr std::array<std::size_t, 5> contexts{32, 64, 128, 256, 400};
  for (const std::size_t prefix : prefixes) {
    for (const std::string& domain : domains) {
      for (const PrefixExample example : selected_prefix_examples(split, domain, prefix)) {
        const Document& document{split[example.document]};
        for (const std::size_t available : contexts) {
          if (available > prefix || (prefix != 256 && prefix != 400 && available != prefix)) {
            continue;
          }
          const std::size_t begin{example.start + prefix - available};
          const auto sequence{
              std::span{document.tokens}.subspan(begin, available + continuation_targets)};
          const Score result{score(model, sequence, available, device)};
          prefix_output << csv(split_name) << ',' << prefix << ',' << available << ','
                        << example.document << ',' << csv(document.metadata.id) << ','
                        << example.start << ',' << csv(document.metadata.domain) << ','
                        << csv(document.metadata.source) << ',' << result.targets << ','
                        << std::setprecision(17) << result.loss_sum << ','
                        << result.loss_sum / static_cast<double>(result.targets) << '\n';
          prefix_output.flush();
        }
      }
    }
    std::println("prefix_group={} complete", prefix);
  }
  std::println("split={} documents={} chunks={} targets={} token_weighted_ce={:.8f}", split_name,
               split.size(), evaluated_chunks, global.targets,
               global.loss_sum / static_cast<double>(global.targets));
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_evaluate_autocomplete: " << error.what() << '\n';
    return 1;
  }
}
