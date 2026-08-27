import leda;
import spar;
import spar.loss.cross_entropy;
import std;

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments final {
  std::filesystem::path spec_path;
  std::filesystem::path run_directory;
  std::uint64_t stop_steps;
  enum class Disposition { Fresh, Branch, Resume, Validate } disposition;
};

enum class IteratorPolicy { Preserve, ScaleStride };

struct ContinuationSource final {
  std::filesystem::path checkpoint;
  std::string checkpoint_sha256;
  std::filesystem::path state;
  std::string state_sha256;
  std::filesystem::path run_spec;
  std::string run_spec_sha256;
  std::size_t sequence_length;
  std::size_t stride;
  std::uint64_t global_step;
  std::uint64_t tokens_seen;
  IteratorPolicy iterator_policy;
};

struct RunSpec final {
  std::string raw;
  std::string spar_commit;
  std::string leda_commit;
  std::string corpus_fingerprint;
  std::string tokenizer_sha256;
  std::string run_spec_payload_sha256;
  std::filesystem::path train_manifest;
  std::string train_manifest_sha256;
  std::filesystem::path validation_manifest;
  std::string validation_manifest_sha256;
  std::size_t tokenizer_vocab_size;
  std::size_t model_vocab_size;
  std::uint64_t seed;
  leda::LedaConfig model;
  std::uint64_t parameters;
  leda::PretrainingConfig training;
  spar::MatmulPrecision precision;
  std::uint64_t maximum_steps;
  std::uint64_t intended_tokens;
  std::vector<std::uint64_t> validation_steps;
  std::vector<std::uint64_t> checkpoint_steps;
  bool save_final;
  std::optional<ContinuationSource> continuation;
};

struct RestoredState final {
  spar::data::BatchIteratorState iterator;
  std::uint64_t global_step;
  std::uint64_t tokens_seen;
  std::uint64_t next_batch_hash;
  double cumulative_update_ms;
};

[[noreturn]] void usage() {
  throw std::invalid_argument{"usage: leda_pretrain_sharded RUN_SPEC RUN_DIR "
                              "STOP_STEPS (fresh|branch|resume|validate)"};
}

std::uint64_t parse_u64(std::string_view text, std::string_view name, bool allow_zero = false) {
  std::uint64_t value{};
  const auto parsed{std::from_chars(text.data(), text.data() + text.size(), value)};
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      (!allow_zero && value == 0)) {
    throw std::invalid_argument{std::string{name} + " must be a valid uint64"};
  }
  return value;
}

std::size_t parse_size(std::string_view text, std::string_view name) {
  const std::uint64_t value{parse_u64(text, name)};
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{std::string{name} + " exceeds size_t"};
  }
  return static_cast<std::size_t>(value);
}

double parse_double(std::string_view text, std::string_view name, bool positive = false) {
  double value{};
  std::istringstream stream{std::string{text}};
  stream >> std::noskipws >> value;
  if (!stream || stream.peek() != std::char_traits<char>::eof() || !std::isfinite(value) ||
      (positive && value <= 0.0)) {
    throw std::invalid_argument{std::string{name} + " must be finite"};
  }
  return value;
}

bool parse_bool(std::string_view text, std::string_view name) {
  if (text == "0") {
    return false;
  }
  if (text == "1") {
    return true;
  }
  throw std::invalid_argument{std::string{name} + " must be 0 or 1"};
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream stream{path, std::ios::binary};
  if (!stream) {
    throw std::runtime_error{"Unable to open file: " + path.string()};
  }
  return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

std::vector<std::uint64_t> parse_steps(std::string_view text, std::string_view name) {
  if (text == "none") {
    return {};
  }
  std::vector<std::uint64_t> result;
  while (!text.empty()) {
    const std::size_t separator{text.find(',')};
    const std::string_view item{text.substr(0, separator)};
    result.push_back(parse_u64(item, name, true));
    if (separator == std::string_view::npos) {
      break;
    }
    text.remove_prefix(separator + 1);
  }
  if (!std::ranges::is_sorted(result) || std::ranges::adjacent_find(result) != result.end()) {
    throw std::invalid_argument{std::string{name} + " must be strictly increasing"};
  }
  return result;
}

bool is_sha256(std::string_view value) {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

RunSpec load_spec(const std::filesystem::path& path) {
  const std::string raw{read_file(path)};
  std::map<std::string, std::string> fields;
  std::istringstream lines{raw};
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty()) {
      continue;
    }
    const std::size_t equals{line.find('=')};
    if (equals == std::string::npos || equals == 0 || equals + 1 == line.size() ||
        !fields.emplace(line.substr(0, equals), line.substr(equals + 1)).second) {
      throw std::invalid_argument{"Invalid or duplicate run-spec field"};
    }
  }
  const auto value = [&](std::string_view key) -> const std::string& {
    const auto found{fields.find(std::string{key})};
    if (found == fields.end()) {
      throw std::invalid_argument{"Missing run-spec field: " + std::string{key}};
    }
    return found->second;
  };
  const std::string& format{value("format")};
  if (format != "LEDA_SCALE_RUN_V1" && format != "LEDA_PRETRAIN_RUN_V2" &&
      format != "LEDA_CONTINUATION_RUN_V3") {
    throw std::invalid_argument{"Unsupported Leda run-spec format"};
  }
  const auto optional_value = [&](std::string_view key, std::string fallback) -> std::string {
    const auto found{fields.find(std::string{key})};
    return found == fields.end() ? std::move(fallback) : found->second;
  };
  const std::size_t vocab{parse_size(value("model_vocab_size"), "model_vocab_size")};
  const std::uint64_t schedule{parse_u64(value("schedule_steps"), "schedule_steps")};
  const std::string& precision{value("precision")};
  if (precision != "full" && precision != "fp16") {
    throw std::invalid_argument{"precision must be full or fp16"};
  }
  RunSpec result{
      .raw = raw,
      .spar_commit = value("spar_commit"),
      .leda_commit = value("leda_commit"),
      .corpus_fingerprint = optional_value("corpus_fingerprint", "legacy"),
      .tokenizer_sha256 = optional_value("tokenizer_sha256", "legacy"),
      .run_spec_payload_sha256 = optional_value("run_spec_payload_sha256", "legacy"),
      .train_manifest = value("train_manifest"),
      .train_manifest_sha256 = value("train_manifest_sha256"),
      .validation_manifest = value("validation_manifest"),
      .validation_manifest_sha256 = value("validation_manifest_sha256"),
      .tokenizer_vocab_size = parse_size(value("tokenizer_vocab_size"), "tokenizer_vocab_size"),
      .model_vocab_size = vocab,
      .seed = parse_u64(value("seed"), "seed", true),
      .model = {.vocab_size = vocab,
                .model_dim = parse_size(value("model_dim"), "model_dim"),
                .hidden_dim = parse_size(value("hidden_dim"), "hidden_dim"),
                .num_layers = parse_size(value("num_layers"), "num_layers"),
                .num_query_heads = parse_size(value("num_query_heads"), "num_query_heads"),
                .num_kv_heads = parse_size(value("num_kv_heads"), "num_kv_heads")},
      .parameters = parse_u64(value("parameters"), "parameters"),
      .training = {.sequence_length = parse_size(value("sequence_length"), "sequence_length"),
                   .stride = parse_size(value("stride"), "stride"),
                   .microbatch_size = parse_size(value("microbatch_size"), "microbatch_size"),
                   .accumulation_steps =
                       parse_size(value("accumulation_steps"), "accumulation_steps"),
                   .max_grad_norm = parse_double(value("max_grad_norm"), "max_grad_norm"),
                   .learning_rate = {.peak_learning_rate = parse_double(value("peak_learning_rate"),
                                                                        "peak_learning_rate", true),
                                     .minimum_learning_rate = parse_double(
                                         value("minimum_learning_rate"), "minimum_learning_rate"),
                                     .warmup_steps =
                                         parse_u64(value("warmup_steps"), "warmup_steps", true),
                                     .decay_steps = schedule},
                   .beta1 = parse_double(value("beta1"), "beta1"),
                   .beta2 = parse_double(value("beta2"), "beta2"),
                   .epsilon = parse_double(value("epsilon"), "epsilon", true),
                   .weight_decay = parse_double(value("weight_decay"), "weight_decay"),
                   .shuffle_seed = parse_u64(value("shuffle_seed"), "shuffle_seed", true)},
      .precision =
          precision == "full" ? spar::MatmulPrecision::Full : spar::MatmulPrecision::Float16Inputs,
      .maximum_steps = parse_u64(value("maximum_steps"), "maximum_steps"),
      .intended_tokens = parse_u64(value("intended_tokens"), "intended_tokens"),
      .validation_steps = parse_steps(value("validation_steps"), "validation_steps"),
      .checkpoint_steps = parse_steps(value("checkpoint_steps"), "checkpoint_steps"),
      .save_final = parse_bool(value("save_final"), "save_final"),
      .continuation = std::nullopt};
  if ((format == "LEDA_PRETRAIN_RUN_V2" || format == "LEDA_CONTINUATION_RUN_V3") &&
      (!is_sha256(result.corpus_fingerprint) || !is_sha256(result.tokenizer_sha256) ||
       !is_sha256(result.run_spec_payload_sha256) || !is_sha256(result.train_manifest_sha256) ||
       !is_sha256(result.validation_manifest_sha256))) {
    throw std::invalid_argument{"Run identity fields must be lowercase SHA-256 digests"};
  }
  if (format == "LEDA_CONTINUATION_RUN_V3") {
    const std::string& policy{value("iterator_policy")};
    if (policy != "preserve" && policy != "scale_stride") {
      throw std::invalid_argument{"iterator_policy must be preserve or scale_stride"};
    }
    result.continuation = ContinuationSource{
        .checkpoint = value("base_checkpoint"),
        .checkpoint_sha256 = value("base_checkpoint_sha256"),
        .state = value("base_state"),
        .state_sha256 = value("base_state_sha256"),
        .run_spec = value("base_run_spec"),
        .run_spec_sha256 = value("base_run_spec_sha256"),
        .sequence_length = parse_size(value("base_sequence_length"), "base_sequence_length"),
        .stride = parse_size(value("base_stride"), "base_stride"),
        .global_step = parse_u64(value("base_global_step"), "base_global_step", true),
        .tokens_seen = parse_u64(value("base_tokens_seen"), "base_tokens_seen", true),
        .iterator_policy =
            policy == "preserve" ? IteratorPolicy::Preserve : IteratorPolicy::ScaleStride};
    if (!is_sha256(result.continuation->checkpoint_sha256) ||
        !is_sha256(result.continuation->state_sha256) ||
        !is_sha256(result.continuation->run_spec_sha256)) {
      throw std::invalid_argument{"Base artifact fields must be lowercase SHA-256 digests"};
    }
  }
  static_cast<void>(leda::decoder_config(result.model));
  leda::validate_pretraining_config(result.training);
  const std::uint64_t targets_per_step{
      static_cast<std::uint64_t>(result.training.microbatch_size) *
      static_cast<std::uint64_t>(result.training.accumulation_steps) *
      static_cast<std::uint64_t>(result.training.sequence_length - 1)};
  const std::uint64_t first_step{result.continuation ? result.continuation->global_step : 0};
  const std::uint64_t first_tokens{result.continuation ? result.continuation->tokens_seen : 0};
  if (result.maximum_steps < first_step ||
      result.maximum_steps - first_step >
          (std::numeric_limits<std::uint64_t>::max() - first_tokens) / targets_per_step ||
      first_tokens + (result.maximum_steps - first_step) * targets_per_step !=
          result.intended_tokens) {
    throw std::invalid_argument{"intended_tokens does not match the configured maximum run"};
  }
  if ((!result.validation_steps.empty() &&
       (result.validation_steps.front() < first_step ||
        result.validation_steps.back() > result.maximum_steps)) ||
      (!result.checkpoint_steps.empty() &&
       (result.checkpoint_steps.front() < first_step ||
        result.checkpoint_steps.back() > result.maximum_steps))) {
    throw std::invalid_argument{"validation/checkpoint step is outside the configured run"};
  }
  return result;
}

Arguments arguments(int argc, char** argv) {
  if (argc != 5) {
    usage();
  }
  const std::string_view disposition{argv[4]};
  if (disposition != "fresh" && disposition != "branch" && disposition != "resume" &&
      disposition != "validate") {
    usage();
  }
  const Arguments::Disposition parsed{disposition == "fresh"    ? Arguments::Disposition::Fresh
                                      : disposition == "branch" ? Arguments::Disposition::Branch
                                      : disposition == "resume" ? Arguments::Disposition::Resume
                                                                : Arguments::Disposition::Validate};
  return {.spec_path = argv[1],
          .run_directory = argv[2],
          .stop_steps = parse_u64(argv[3], "STOP_STEPS"),
          .disposition = parsed};
}

std::vector<std::filesystem::path> paths(const std::filesystem::path& manifest) {
  std::ifstream stream{manifest};
  if (!stream) {
    throw std::runtime_error{"Unable to open shard manifest: " + manifest.string()};
  }
  std::vector<std::filesystem::path> result;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      std::filesystem::path path{line};
      result.push_back(path.is_absolute() ? std::move(path) : manifest.parent_path() / path);
    }
  }
  if (!stream.eof() || result.empty()) {
    throw std::runtime_error{"Shard manifest is unreadable or empty: " + manifest.string()};
  }
  return result;
}

std::string precision_name(spar::MatmulPrecision precision) {
  return precision == spar::MatmulPrecision::Full ? "full" : "fp16";
}

std::optional<std::uint64_t> resident_kib() {
  std::ifstream stream{"/proc/self/status"};
  std::string key;
  while (stream >> key) {
    if (key == "VmRSS:") {
      std::uint64_t value{};
      stream >> value;
      return value;
    }
    std::string ignored;
    std::getline(stream, ignored);
  }
  return std::nullopt;
}

void write_atomic(const std::filesystem::path& path, std::string_view contents) {
  const std::filesystem::path temporary{path.string() + ".tmp"};
  {
    std::ofstream stream{temporary, std::ios::binary | std::ios::trunc};
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.flush();
  }
  std::filesystem::rename(temporary, path);
}

std::string latest_step(const std::filesystem::path& directory) {
  std::ifstream stream{directory / "latest.txt"};
  std::string step;
  stream >> step;
  if (!stream || step.empty() ||
      !std::ranges::all_of(step, [](char value) { return value >= '0' && value <= '9'; })) {
    throw std::runtime_error{"Missing or invalid latest run-state pointer"};
  }
  return step;
}

std::uint64_t batch_hash(const spar::Tensor& batch) {
  if (batch.dtype() != spar::DType::Int64 || !batch.device().is_cpu()) {
    throw std::logic_error{"Batch identity requires a CPU Int64 batch"};
  }
  std::uint64_t hash{1469598103934665603ULL};
  for (const std::int64_t value : batch.span<std::int64_t>()) {
    const std::uint64_t bits{static_cast<std::uint64_t>(value)};
    for (unsigned shift{}; shift < 64; shift += 8) {
      hash ^= (bits >> shift) & 0xffU;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::uint64_t next_batch_hash(const spar::data::ShardedWindowDataset& dataset,
                              spar::data::BatchConfig config,
                              spar::data::BatchIteratorState state) {
  spar::data::LMBatchIterator probe{dataset, config};
  probe.set_state(state);
  auto batch{probe.next_batch()};
  if (!batch) {
    probe.next_epoch();
    batch = probe.next_batch();
  }
  if (!batch) {
    throw std::logic_error{"Training dataset contains no next batch"};
  }
  return batch_hash(*batch);
}

RestoredState restore_state(const std::filesystem::path& path) {
  std::ifstream stream{path};
  std::string magic;
  std::uint64_t version{};
  RestoredState result{};
  stream >> magic >> version >> result.iterator.epoch >> result.iterator.cursor >>
      result.global_step >> result.tokens_seen >> result.next_batch_hash;
  if (version == 4) {
    stream >> result.cumulative_update_ms;
  }
  if (!stream || magic != "LEDA_RUN_STATE" || (version != 3 && version != 4)) {
    throw std::runtime_error{"Invalid Leda run-state sidecar"};
  }
  return result;
}

void save_boundary(const std::filesystem::path& directory, leda::Leda& model,
                   spar::optim::AdamW& optimizer, const spar::Random& random,
                   spar::checkpoint::TrainingProgress progress,
                   spar::data::BatchIteratorState iterator_state,
                   const spar::data::ShardedWindowDataset& dataset,
                   spar::data::BatchConfig batch_config, spar::Device cuda,
                   double cumulative_update_ms) {
  const std::string step{std::to_string(progress.global_step)};
  const std::uint64_t next_hash{next_batch_hash(dataset, batch_config, iterator_state)};
  optimizer.zero_grad();
  optimizer.move_to(spar::Device::cpu());
  spar::checkpoint::save_training_checkpoint(directory / ("checkpoint-" + step + ".sparckpt"),
                                             model.decoder(), optimizer, random, progress);
  std::ostringstream state;
  state << "LEDA_RUN_STATE 4\n"
        << iterator_state.epoch << ' ' << iterator_state.cursor << '\n'
        << progress.global_step << ' ' << progress.tokens_seen << '\n'
        << next_hash << '\n'
        << std::setprecision(17) << cumulative_update_ms << '\n';
  write_atomic(directory / ("state-" + step + ".txt"), state.str());
  write_atomic(directory / "latest.txt", step + "\n");
  optimizer.move_to(cuda);
}

struct ValidationResult final {
  double mean_loss;
  std::size_t windows;
};

ValidationResult evaluate_windows(const leda::Leda& model,
                                  const spar::data::ShardedWindowDataset& dataset,
                                  std::size_t batch_size, std::uint64_t step,
                                  const std::filesystem::path& directory) {
  spar::data::LMBatchIterator batches{dataset,
                                      {.batch_size = batch_size,
                                       .token_dtype = spar::DType::Int64,
                                       .shuffle = false,
                                       .shuffle_seed = 0,
                                       .drop_last = false}};
  std::ofstream distribution{directory / ("validation-" + std::to_string(step) + ".csv")};
  distribution << "window,mean_loss\n";
  double total{};
  std::uint64_t targets{};
  std::size_t window{};
  const spar::Device device{leda::parameters(const_cast<leda::Leda&>(model))[0].tensor().device()};
  while (auto batch{batches.next_batch()}) {
    const spar::Tensor device_batch{batch->to(device)};
    const spar::Tensor losses{spar::loss::language_model_cross_entropy(
        model.forward(device_batch), device_batch, spar::loss::Reduction::None)};
    const spar::Tensor host{losses.to(spar::Device::cpu())};
    const std::size_t rows{static_cast<std::size_t>(host.shape()[0])};
    const std::size_t width{static_cast<std::size_t>(host.shape()[1])};
    const auto values{host.span<float>()};
    for (std::size_t row{}; row < rows; ++row) {
      const double sum{
          std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(row * width),
                          values.begin() + static_cast<std::ptrdiff_t>((row + 1) * width), 0.0)};
      distribution << window++ << ',' << sum / static_cast<double>(width) << '\n';
      total += sum;
      targets += width;
    }
  }
  if (targets == 0) {
    throw std::runtime_error{"Validation corpus contains no targets"};
  }
  const double mean_loss{total / static_cast<double>(targets)};
  if (!std::isfinite(mean_loss)) {
    throw std::runtime_error{"Validation produced non-finite loss"};
  }
  return {mean_loss, window};
}

bool contains(std::span<const std::uint64_t> steps, std::uint64_t step) {
  return std::ranges::binary_search(steps, step);
}

int run(int argc, char** argv) {
  const Arguments args{arguments(argc, argv)};
  const RunSpec spec{load_spec(args.spec_path)};
  const bool branch{args.disposition == Arguments::Disposition::Branch};
  const bool resume{args.disposition == Arguments::Disposition::Resume};
  const bool validate{args.disposition == Arguments::Disposition::Validate};
  if (branch != spec.continuation.has_value()) {
    if (branch) {
      throw std::invalid_argument{"branch requires a LEDA_CONTINUATION_RUN_V3 spec"};
    }
    if (!resume && !validate) {
      throw std::invalid_argument{"A continuation spec must begin with branch, not fresh"};
    }
  }
  if (args.stop_steps > spec.maximum_steps) {
    throw std::invalid_argument{"STOP_STEPS exceeds the immutable maximum_steps"};
  }
  if (validate) {
    // Validation performs no experiment-directory writes.
  } else if (resume) {
    if (read_file(args.run_directory / "run-spec.txt") != spec.raw) {
      throw std::invalid_argument{"Requested run spec conflicts with the stored immutable spec"};
    }
  } else {
    std::filesystem::create_directories(args.run_directory);
    if (std::filesystem::exists(args.run_directory / "run-spec.txt")) {
      throw std::invalid_argument{"Fresh run directory already contains a run spec"};
    }
    write_atomic(args.run_directory / "run-spec.txt", spec.raw);
  }

  const auto started{Clock::now()};
  const spar::data::ShardedTokenCorpus train_corpus{paths(spec.train_manifest)};
  const spar::data::ShardedTokenCorpus validation_corpus{paths(spec.validation_manifest)};
  if (train_corpus.tokenizer_vocab_size() != spec.tokenizer_vocab_size ||
      train_corpus.model_vocab_size() != spec.model_vocab_size ||
      validation_corpus.model_vocab_size() != spec.model_vocab_size ||
      validation_corpus.eod_token_id() != train_corpus.eod_token_id()) {
    throw std::invalid_argument{"Corpus metadata conflicts with the immutable run spec"};
  }
  const spar::data::ShardedWindowDataset train_dataset{
      train_corpus, {spec.training.sequence_length, spec.training.stride}};
  const spar::data::ShardedWindowDataset validation_dataset{
      validation_corpus, {spec.training.sequence_length, spec.training.stride}};
  const spar::data::BatchConfig batch_config{.batch_size = spec.training.microbatch_size,
                                             .token_dtype = spar::DType::Int64,
                                             .shuffle = true,
                                             .shuffle_seed = spec.training.shuffle_seed,
                                             .drop_last = false};
  spar::data::LMBatchIterator batches{train_dataset, batch_config};
  spar::Random random{spec.seed};
  leda::Leda model{spec.model, random};
  const auto statistics{leda::model_statistics(model)};
  if (statistics.total_parameters != spec.parameters) {
    throw std::invalid_argument{"Programmatic Parameter count conflicts with the run spec"};
  }
  if (validate) {
    const auto memory{leda::adamw_memory_estimate(model)};
    std::println("validated parameters={} persistent_bytes={} train_tokens={} "
                 "validation_tokens={} windows={} intended_tokens={} "
                 "precision={} corpus={} tokenizer={} run_spec={}",
                 statistics.total_parameters, memory.persistent_training_bytes,
                 train_corpus.token_count(), validation_corpus.token_count(),
                 train_dataset.window_count(), spec.intended_tokens, precision_name(spec.precision),
                 spec.corpus_fingerprint, spec.tokenizer_sha256, spec.run_spec_payload_sha256);
    return 0;
  }
  spar::optim::AdamW optimizer{
      leda::parameters(model), spec.training.learning_rate.peak_learning_rate,
      spec.training.beta1,     spec.training.beta2,
      spec.training.epsilon,   spec.training.weight_decay};
  spar::checkpoint::TrainingProgress progress{};
  std::uint64_t restored_hash{};
  double cumulative_update_ms{};
  if (branch || resume) {
    const std::string step{resume ? latest_step(args.run_directory) : std::string{}};
    const std::filesystem::path checkpoint{resume ? args.run_directory /
                                                        ("checkpoint-" + step + ".sparckpt")
                                                  : spec.continuation->checkpoint};
    const std::filesystem::path state{resume ? args.run_directory / ("state-" + step + ".txt")
                                             : spec.continuation->state};
    auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
    const RestoredState sidecar{restore_state(state)};
    if (loaded.progress.global_step != sidecar.global_step ||
        loaded.progress.tokens_seen != sidecar.tokens_seen) {
      throw std::runtime_error{"Checkpoint and iterator sidecar progress disagree"};
    }
    if (branch && (sidecar.global_step != spec.continuation->global_step ||
                   sidecar.tokens_seen != spec.continuation->tokens_seen)) {
      throw std::runtime_error{"Base progress conflicts with the immutable continuation spec"};
    }
    model = leda::Leda::from_decoder(spec.model, std::move(loaded.model));
    optimizer = std::move(loaded.optimizer);
    random = std::move(loaded.random);
    progress = loaded.progress;
    if (optimizer.beta1() != spec.training.beta1 || optimizer.beta2() != spec.training.beta2 ||
        optimizer.epsilon() != spec.training.epsilon ||
        optimizer.weight_decay() != spec.training.weight_decay) {
      throw std::runtime_error{"Base AdamW hyperparameters conflict with the continuation spec"};
    }
    spar::data::BatchIteratorState iterator{sidecar.iterator};
    if (branch && spec.continuation->iterator_policy == IteratorPolicy::ScaleStride) {
      if ((iterator.cursor != 0 &&
           spec.continuation->stride >
               std::numeric_limits<std::uint64_t>::max() / iterator.cursor) ||
          iterator.cursor * spec.continuation->stride % spec.training.stride != 0) {
        throw std::runtime_error{"Base iterator cursor cannot be scaled exactly to the new stride"};
      }
      iterator.cursor = iterator.cursor * spec.continuation->stride / spec.training.stride;
    } else if (branch && (spec.continuation->sequence_length != spec.training.sequence_length ||
                          spec.continuation->stride != spec.training.stride)) {
      throw std::runtime_error{"preserve iterator policy requires unchanged sequence and stride"};
    }
    batches.set_state(iterator);
    restored_hash = next_batch_hash(train_dataset, batch_config, iterator);
    if ((!branch || spec.continuation->iterator_policy == IteratorPolicy::Preserve) &&
        restored_hash != sidecar.next_batch_hash) {
      throw std::runtime_error{"First restored batch identity does not match the sidecar"};
    }
    cumulative_update_ms = resume ? sidecar.cumulative_update_ms : 0.0;
    std::println("{} step={} tokens={} epoch={} cursor={} next_batch_hash={}",
                 branch ? "branched" : "resumed", progress.global_step, progress.tokens_seen,
                 iterator.epoch, iterator.cursor, restored_hash);
  }
  if (args.stop_steps < progress.global_step) {
    throw std::invalid_argument{"STOP_STEPS precedes the restored global_step"};
  }
  const spar::Device cuda{spar::Device::cuda(0)};
  optimizer.move_to(cuda);
  spar::set_matmul_precision(cuda, spec.precision);
  const auto ready{Clock::now()};
  std::ofstream log{args.run_directory / "metrics.csv", resume ? std::ios::app : std::ios::trunc};
  if (!resume) {
    log << "step,tokens_seen,train_mean_loss,validation_mean_loss,learning_"
           "rate,gradient_norm,"
           "clip_scale,clipped,update_ms,tokens_per_second,validation_ms,host_"
           "rss_kib,precision,cumulative_update_s,checkpoint_id,corpus_"
           "fingerprint,tokenizer_sha256,run_spec_payload_sha256\n";
  }
  const auto log_row = [&](double train_loss, double validation_loss, double learning_rate,
                           double gradient_norm, double clip_scale, bool clipped, double update_ms,
                           double validation_ms, std::string_view checkpoint_id) {
    const double token_rate{
        update_ms > 0.0
            ? static_cast<double>(spec.training.microbatch_size * spec.training.accumulation_steps *
                                  (spec.training.sequence_length - 1)) /
                  (update_ms / 1000.0)
            : std::numeric_limits<double>::quiet_NaN()};
    log << progress.global_step << ',' << progress.tokens_seen << ',' << train_loss << ','
        << validation_loss << ',' << learning_rate << ',' << gradient_norm << ',' << clip_scale
        << ',' << (clipped ? 1 : 0) << ',' << update_ms << ',' << token_rate << ',' << validation_ms
        << ',' << resident_kib().value_or(0) << ',' << precision_name(spec.precision) << ','
        << cumulative_update_ms / 1000.0 << ',' << checkpoint_id << ',' << spec.corpus_fingerprint
        << ',' << spec.tokenizer_sha256 << ',' << spec.run_spec_payload_sha256 << '\n';
    log.flush();
    std::println("step={} tokens={} loss={:.6f} validation={:.6f} lr={:.8f} grad={:.6f} "
                 "clip={:.6f} ms={:.2f} tok_s={:.1f}",
                 progress.global_step, progress.tokens_seen, train_loss, validation_loss,
                 learning_rate, gradient_norm, clip_scale, update_ms, token_rate);
  };
  if (!resume && contains(spec.validation_steps, progress.global_step)) {
    const auto validation_started{Clock::now()};
    const ValidationResult validation{evaluate_windows(model, validation_dataset,
                                                       spec.training.microbatch_size,
                                                       progress.global_step, args.run_directory)};
    const double validation_ms{
        std::chrono::duration<double, std::milli>(Clock::now() - validation_started).count()};
    log_row(std::numeric_limits<double>::quiet_NaN(), validation.mean_loss, 0.0,
            std::numeric_limits<double>::quiet_NaN(), 1.0, false, 0.0, validation_ms, "none");
  }
  while (progress.global_step < args.stop_steps) {
    const auto update_started{Clock::now()};
    auto result{leda::train_update(model, optimizer, batches, progress, spec.training)};
    if (!result) {
      batches.next_epoch();
      result = leda::train_update(model, optimizer, batches, progress, spec.training);
    }
    if (!result || !std::isfinite(result->mean_loss) || !std::isfinite(result->grad_norm) ||
        !std::isfinite(result->clip_scale) || !std::isfinite(result->learning_rate)) {
      throw std::runtime_error{"Non-finite or missing Leda training update"};
    }
    const double update_ms{
        std::chrono::duration<double, std::milli>(Clock::now() - update_started).count()};
    cumulative_update_ms += update_ms;
    double validation_loss{std::numeric_limits<double>::quiet_NaN()};
    double validation_ms{};
    if (contains(spec.validation_steps, progress.global_step)) {
      const auto validation_started{Clock::now()};
      validation_loss = evaluate_windows(model, validation_dataset, spec.training.microbatch_size,
                                         progress.global_step, args.run_directory)
                            .mean_loss;
      validation_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - validation_started).count();
    }
    const bool checkpoint_boundary{contains(spec.checkpoint_steps, progress.global_step) ||
                                   (spec.save_final && progress.global_step == args.stop_steps)};
    const std::string checkpoint_id{
        checkpoint_boundary ? "checkpoint-" + std::to_string(progress.global_step) : "none"};
    log_row(result->mean_loss, validation_loss, result->learning_rate, result->grad_norm,
            result->clip_scale, result->clipped, update_ms, validation_ms, checkpoint_id);
    if (checkpoint_boundary) {
      save_boundary(args.run_directory, model, optimizer, random, progress, batches.state(),
                    train_dataset, batch_config, cuda, cumulative_update_ms);
      spar::set_matmul_precision(cuda, spec.precision);
    }
  }
  std::println("complete steps={} tokens={} elapsed_s={:.3f} parameters={} "
               "setup_s={:.3f}",
               progress.global_step, progress.tokens_seen,
               std::chrono::duration<double>(Clock::now() - ready).count(),
               statistics.total_parameters, std::chrono::duration<double>(ready - started).count());
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_pretrain_sharded: " << error.what() << '\n';
    return 1;
  }
}
