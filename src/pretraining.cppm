export module leda.pretraining;

import std;
export import leda.model;
export import spar.checkpoint;
export import spar.data.batch_iterator;
export import spar.optim.adamw;
export import spar.training;

export namespace leda {

struct PretrainingConfig final {
  std::size_t sequence_length;
  std::size_t stride;
  std::size_t microbatch_size;
  std::size_t accumulation_steps;
  double max_grad_norm;
  spar::training::WarmupCosineConfig learning_rate;
  double beta1{0.9};
  double beta2{0.95};
  double epsilon{1.0e-8};
  double weight_decay{0.1};
  std::uint64_t shuffle_seed;
};

void validate_pretraining_config(const PretrainingConfig& config);

struct TrainingStepResult final {
  double mean_loss;
  std::uint64_t target_count;
  std::size_t microbatches;
  double grad_norm;
  double clip_scale;
  bool clipped;
  double learning_rate;
};

/// Executes one Leda optimizer update. Exceptions are fatal for the current run: consumed batches,
/// tokens_seen, and partial gradients are not transactionally rolled back. The optimizer must track
/// exactly the model's unique Parameters; ownership is checked before gradients or data are
/// mutated.
[[nodiscard]] std::optional<TrainingStepResult>
train_update(Leda& model, spar::optim::AdamW& optimizer, spar::data::LMBatchIterator& batches,
             spar::checkpoint::TrainingProgress& progress, const PretrainingConfig& config);

struct EvaluationResult final {
  double mean_loss;
  std::uint64_t target_count;
  std::size_t batches;
};

/// Computes one globally target-weighted validation pass without calling backward. Temporary
/// forward graphs are still constructed because Spar has no no-grad mode yet.
[[nodiscard]] EvaluationResult evaluate(const Leda& model,
                                        spar::data::LMBatchIterator& validation_batches);

} // namespace leda
