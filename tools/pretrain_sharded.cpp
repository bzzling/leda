import leda;
import spar;
import std;

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments final {
  std::filesystem::path train_manifest;
  std::filesystem::path validation_manifest;
  std::filesystem::path run_directory;
  std::uint64_t maximum_steps;
  std::uint64_t schedule_steps;
  spar::MatmulPrecision precision;
  bool overfit;
  bool resume;
};

struct RestoredState final {
  spar::data::BatchIteratorState iterator;
  std::uint64_t global_step;
  std::uint64_t tokens_seen;
  std::uint64_t schedule_steps;
  std::string precision;
  std::string mode;
};

[[noreturn]] void usage() {
  throw std::invalid_argument{"usage: leda_pretrain_sharded TRAIN_SHARDS "
                              "VALIDATION_SHARDS RUN_DIR STOP_STEPS "
                              "SCHEDULE_STEPS "
                              "(full|fp16) (train|overfit) (fresh|resume)"};
}

std::uint64_t parse_positive(std::string_view text, std::string_view name) {
  std::uint64_t value{};
  const auto parsed{std::from_chars(text.data(), text.data() + text.size(), value)};
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) {
    throw std::invalid_argument{std::string{name} + " must be a positive uint64"};
  }
  return value;
}

Arguments arguments(int argc, char** argv) {
  if (argc != 9) {
    usage();
  }
  const std::uint64_t maximum_steps{parse_positive(argv[4], "STOP_STEPS")};
  const std::uint64_t schedule_steps{parse_positive(argv[5], "SCHEDULE_STEPS")};
  const std::string_view precision_text{argv[6]};
  const std::string_view mode{argv[7]};
  const std::string_view disposition{argv[8]};
  if (maximum_steps > schedule_steps) {
    throw std::invalid_argument{"STOP_STEPS must not exceed SCHEDULE_STEPS"};
  }
  if (precision_text != "full" && precision_text != "fp16") {
    usage();
  }
  if (mode != "train" && mode != "overfit") {
    usage();
  }
  if (disposition != "fresh" && disposition != "resume") {
    usage();
  }
  return {.train_manifest = argv[1],
          .validation_manifest = argv[2],
          .run_directory = argv[3],
          .maximum_steps = maximum_steps,
          .schedule_steps = schedule_steps,
          .precision = precision_text == "full" ? spar::MatmulPrecision::Full
                                                : spar::MatmulPrecision::Float16Inputs,
          .overfit = mode == "overfit",
          .resume = disposition == "resume"};
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
  const auto temporary{path.string() + ".tmp"};
  {
    std::ofstream stream{temporary, std::ios::binary | std::ios::trunc};
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.flush();
  }
  std::filesystem::rename(temporary, path);
}

void save_boundary(const std::filesystem::path& directory, leda::Leda& model,
                   spar::optim::AdamW& optimizer, const spar::Random& random,
                   spar::checkpoint::TrainingProgress progress,
                   spar::data::BatchIteratorState iterator_state, spar::Device cuda,
                   const Arguments& args) {
  const std::string step{std::to_string(progress.global_step)};
  const auto checkpoint{directory / ("checkpoint-" + step + ".sparckpt")};
  const auto state_path{directory / ("state-" + step + ".txt")};
  optimizer.zero_grad();
  optimizer.move_to(spar::Device::cpu());
  spar::checkpoint::save_training_checkpoint(checkpoint, model.decoder(), optimizer, random,
                                             progress);
  std::ostringstream state;
  state << "LEDA_RUN_STATE 2\n"
        << iterator_state.epoch << ' ' << iterator_state.cursor << '\n'
        << progress.global_step << ' ' << progress.tokens_seen << '\n'
        << args.schedule_steps << ' ' << precision_name(args.precision) << ' '
        << (args.overfit ? "overfit" : "train") << '\n';
  write_atomic(state_path, state.str());
  write_atomic(directory / "latest.txt", step + "\n");
  optimizer.move_to(cuda);
}

RestoredState restore_state(const std::filesystem::path& path) {
  std::ifstream stream{path};
  std::string magic;
  std::uint64_t version{};
  RestoredState result{};
  stream >> magic >> version >> result.iterator.epoch >> result.iterator.cursor >>
      result.global_step >> result.tokens_seen >> result.schedule_steps >> result.precision >>
      result.mode;
  if (!stream || magic != "LEDA_RUN_STATE" || version != 2) {
    throw std::runtime_error{"Invalid Leda run-state sidecar"};
  }
  return result;
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

leda::PretrainingConfig recipe(const Arguments& args) {
  const double peak{args.overfit ? 2.0e-3 : 3.0e-4};
  return {.sequence_length = 128,
          .stride = 128,
          .microbatch_size = 4,
          .accumulation_steps = 4,
          .max_grad_norm = 1.0,
          .learning_rate = {.peak_learning_rate = peak,
                            .minimum_learning_rate = peak * 0.1,
                            .warmup_steps = args.overfit ? 10U : 100U,
                            .decay_steps = args.schedule_steps},
          .beta1 = 0.9,
          .beta2 = 0.95,
          .epsilon = 1.0e-8,
          .weight_decay = args.overfit ? 0.0 : 0.1,
          .shuffle_seed = 2802};
}

int run(int argc, char** argv) {
  const Arguments args{arguments(argc, argv)};
  std::filesystem::create_directories(args.run_directory);
  const auto train_started{Clock::now()};
  const spar::data::ShardedTokenCorpus train_corpus{paths(args.train_manifest)};
  const spar::data::ShardedTokenCorpus validation_corpus{paths(args.validation_manifest)};
  if (train_corpus.model_vocab_size() != validation_corpus.model_vocab_size() ||
      train_corpus.eod_token_id() != validation_corpus.eod_token_id()) {
    throw std::invalid_argument{"Training and validation corpora have different vocab metadata"};
  }
  const spar::data::ShardedWindowDataset train_dataset{train_corpus, {128, 128}};
  const spar::data::ShardedWindowDataset validation_dataset{validation_corpus, {128, 128}};
  const auto loaded_at{Clock::now()};
  const spar::Device cuda{spar::Device::cuda(0)};
  spar::set_matmul_precision(cuda, args.precision);
  const leda::LedaConfig model_config{leda::leda_tiny(train_corpus.model_vocab_size())};
  spar::Random random{2801};
  spar::checkpoint::TrainingProgress progress{};
  leda::Leda model{model_config, random};
  spar::optim::AdamW optimizer{
      leda::parameters(model), args.overfit ? 2.0e-3 : 3.0e-4, 0.9, 0.95, 1.0e-8,
      args.overfit ? 0.0 : 0.1};
  spar::data::LMBatchIterator batches{train_dataset,
                                      {.batch_size = 4,
                                       .token_dtype = spar::DType::Int64,
                                       .shuffle = !args.overfit,
                                       .shuffle_seed = 2802,
                                       .drop_last = false}};
  if (args.resume) {
    const std::string step{latest_step(args.run_directory)};
    auto loaded{spar::checkpoint::load_training_checkpoint(args.run_directory /
                                                           ("checkpoint-" + step + ".sparckpt"))};
    const RestoredState sidecar{restore_state(args.run_directory / ("state-" + step + ".txt"))};
    if (loaded.progress.global_step != sidecar.global_step ||
        loaded.progress.tokens_seen != sidecar.tokens_seen ||
        sidecar.schedule_steps != args.schedule_steps ||
        sidecar.precision != precision_name(args.precision) ||
        sidecar.mode != (args.overfit ? "overfit" : "train")) {
      throw std::runtime_error{"Checkpoint and iterator sidecar progress do not agree"};
    }
    model = leda::Leda::from_decoder(model_config, std::move(loaded.model));
    optimizer = std::move(loaded.optimizer);
    random = std::move(loaded.random);
    progress = loaded.progress;
    batches.set_state(sidecar.iterator);
    std::println("resumed step={} tokens={} epoch={} cursor={} schedule_steps={}",
                 progress.global_step, progress.tokens_seen, sidecar.iterator.epoch,
                 sidecar.iterator.cursor, args.schedule_steps);
  } else if (std::filesystem::exists(args.run_directory / "latest.txt")) {
    throw std::invalid_argument{"Fresh run directory already contains a checkpoint"};
  }
  optimizer.move_to(cuda);
  spar::set_matmul_precision(cuda, args.precision);
  const auto model_ready{Clock::now()};
  std::ofstream log{args.run_directory / "metrics.csv",
                    args.resume ? std::ios::app : std::ios::trunc};
  log.exceptions(std::ios::badbit | std::ios::failbit);
  if (!args.resume) {
    log << "step,tokens_seen,train_mean_loss,validation_mean_loss,learning_"
           "rate,gradient_norm,"
           "clip_scale,tokens_per_second,ms_per_update,host_rss_kib,"
           "precision\n";
  }
  std::println("corpus_load_s={:.3f} model_setup_s={:.3f} train_shards={} "
               "train_documents={} "
               "train_tokens={} validation_tokens={} train_windows={} host_rss_kib={}",
               std::chrono::duration<double>(loaded_at - train_started).count(),
               std::chrono::duration<double>(model_ready - loaded_at).count(),
               train_corpus.shard_count(), train_corpus.document_count(),
               train_corpus.token_count(), validation_corpus.token_count(),
               train_dataset.window_count(), resident_kib().value_or(0));
  const leda::PretrainingConfig training{recipe(args)};
  constexpr std::uint64_t report_interval{10};
  constexpr std::uint64_t validation_interval{250};
  constexpr std::uint64_t checkpoint_interval{250};
  double loss_sum{};
  std::uint64_t loss_targets{};
  auto report_started{Clock::now()};
  while (progress.global_step < args.maximum_steps) {
    auto result{leda::train_update(model, optimizer, batches, progress, training)};
    if (!result) {
      batches.next_epoch();
      result = leda::train_update(model, optimizer, batches, progress, training);
    }
    if (!result || !std::isfinite(result->mean_loss) || !std::isfinite(result->grad_norm) ||
        !std::isfinite(result->clip_scale)) {
      throw std::runtime_error{"Non-finite or missing Leda training update"};
    }
    loss_sum += result->mean_loss * static_cast<double>(result->target_count);
    loss_targets += result->target_count;
    if (args.overfit) {
      batches.set_state({});
    }
    const bool report{progress.global_step % report_interval == 0 ||
                      progress.global_step == args.maximum_steps};
    const bool validate{progress.global_step % validation_interval == 0 ||
                        progress.global_step == args.maximum_steps};
    double validation_loss{std::numeric_limits<double>::quiet_NaN()};
    if (validate && !args.overfit) {
      spar::data::LMBatchIterator validation{validation_dataset,
                                             {.batch_size = 4,
                                              .token_dtype = spar::DType::Int64,
                                              .shuffle = false,
                                              .shuffle_seed = 0,
                                              .drop_last = false}};
      validation_loss = leda::evaluate(model, validation).mean_loss;
    }
    if (report) {
      spar::synchronize(cuda);
      const auto now{Clock::now()};
      const double seconds{std::chrono::duration<double>(now - report_started).count()};
      const double mean_loss{loss_sum / static_cast<double>(loss_targets)};
      const double tokens_per_second{static_cast<double>(loss_targets) / seconds};
      const double milliseconds_per_update{seconds * 1000.0 / report_interval};
      const auto rss{resident_kib()};
      std::println("step={} tokens={} loss={:.6f} validation={:.6f} lr={:.8f} "
                   "grad_norm={:.6f} clip={:.6f} tok_s={:.1f} ms_update={:.2f} "
                   "rss_kib={}",
                   progress.global_step, progress.tokens_seen, mean_loss, validation_loss,
                   result->learning_rate, result->grad_norm, result->clip_scale, tokens_per_second,
                   milliseconds_per_update, rss.value_or(0));
      log << progress.global_step << ',' << progress.tokens_seen << ',' << mean_loss << ','
          << validation_loss << ',' << result->learning_rate << ',' << result->grad_norm << ','
          << result->clip_scale << ',' << tokens_per_second << ',' << milliseconds_per_update << ','
          << rss.value_or(0) << ',' << precision_name(args.precision) << '\n';
      log.flush();
      loss_sum = 0.0;
      loss_targets = 0;
      report_started = now;
    }
    if (progress.global_step % checkpoint_interval == 0 ||
        progress.global_step == args.maximum_steps) {
      save_boundary(args.run_directory, model, optimizer, random, progress, batches.state(), cuda,
                    args);
      spar::set_matmul_precision(cuda, args.precision);
    }
  }
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
