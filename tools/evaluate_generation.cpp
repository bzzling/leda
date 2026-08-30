import leda;
import spar;
import std;

namespace {

constexpr spar::tokenizer::TokenId eod_token{8192};
constexpr std::size_t maximum_generation{64};
constexpr std::array<std::size_t, 3> report_lengths{16, 32, 64};
constexpr std::array<std::uint64_t, 3> sampling_seeds{3303, 3304, 3305};

struct Prompt final {
  std::string category;
  std::string text;
};

struct Recipe final {
  std::string name;
  bool greedy;
  double temperature;
  std::optional<std::size_t> top_k;
  std::optional<double> top_p;
};

struct Generation final {
  std::vector<spar::tokenizer::TokenId> tokens;
  bool eod{};
};

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> result;
  std::size_t begin{};
  while (true) {
    const std::size_t end{line.find('\t', begin)};
    result.push_back(line.substr(begin, end == std::string_view::npos ? end : end - begin));
    if (end == std::string_view::npos) {
      return result;
    }
    begin = end + 1;
  }
}

std::vector<Prompt> prompts(const std::filesystem::path& path) {
  std::ifstream stream{path};
  std::string line;
  if (!stream || !std::getline(stream, line) || line != "category\tprompt") {
    throw std::invalid_argument{"Invalid scientific prompt TSV"};
  }
  std::vector<Prompt> result;
  while (std::getline(stream, line)) {
    const auto fields{split_tabs(line)};
    if (fields.size() != 2 || fields[0].empty() || fields[1].empty()) {
      throw std::invalid_argument{"Invalid scientific prompt row"};
    }
    result.push_back({std::string{fields[0]}, std::string{fields[1]}});
  }
  if (result.empty()) {
    throw std::invalid_argument{"Scientific prompt suite is empty"};
  }
  return result;
}

std::optional<std::size_t> optional_size(std::string_view value, std::string_view name) {
  if (value == "-") {
    return std::nullopt;
  }
  std::size_t result{};
  const auto parsed{std::from_chars(value.data(), value.data() + value.size(), result)};
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0) {
    throw std::invalid_argument{std::string{name} + " must be '-' or a positive integer"};
  }
  return result;
}

std::optional<double> optional_real(std::string_view value, std::string_view name) {
  if (value == "-") {
    return std::nullopt;
  }
  std::size_t consumed{};
  double result{};
  try {
    result = std::stod(std::string{value}, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument{std::string{name} + " has an invalid value"};
  }
  if (consumed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument{std::string{name} + " has an invalid value"};
  }
  return result;
}

std::vector<Recipe> recipes(const std::filesystem::path& path) {
  std::ifstream stream{path};
  std::string line;
  if (!stream || !std::getline(stream, line) || line != "name\tgreedy\ttemperature\ttop_k\ttop_p") {
    throw std::invalid_argument{"Invalid decoding recipe TSV"};
  }
  std::vector<Recipe> result;
  std::set<std::string> names;
  while (std::getline(stream, line)) {
    const auto fields{split_tabs(line)};
    if (fields.size() != 5 || !names.emplace(fields[0]).second ||
        (fields[1] != "0" && fields[1] != "1")) {
      throw std::invalid_argument{"Invalid decoding recipe row"};
    }
    const bool greedy{fields[1] == "1"};
    const auto temperature{optional_real(fields[2], "temperature")};
    if ((!greedy && (!temperature || *temperature <= 0.0)) || (greedy && temperature)) {
      throw std::invalid_argument{"Greedy and sampled recipe fields are inconsistent"};
    }
    result.push_back({std::string{fields[0]}, greedy, temperature.value_or(1.0),
                      optional_size(fields[3], "top_k"), optional_real(fields[4], "top_p")});
  }
  if (result.empty()) {
    throw std::invalid_argument{"Decoding recipe TSV is empty"};
  }
  return result;
}

spar::Tensor token_tensor(std::span<const spar::tokenizer::TokenId> tokens,
                          const spar::Device& device) {
  spar::Tensor host{spar::Shape{1, static_cast<std::int64_t>(tokens.size())}, spar::DType::Int64};
  std::ranges::transform(tokens, host.span<std::int64_t>().begin(),
                         [](auto token) { return static_cast<std::int64_t>(token); });
  return host.to(device);
}

std::string hexadecimal(std::string_view bytes) {
  constexpr char digits[]{"0123456789abcdef"};
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const char character : bytes) {
    const auto byte{static_cast<unsigned char>(character)};
    result += digits[byte >> 4U];
    result += digits[byte & 0x0fU];
  }
  return result;
}

std::string token_list(std::span<const spar::tokenizer::TokenId> tokens) {
  std::ostringstream result;
  for (std::size_t index{}; index < tokens.size(); ++index) {
    result << (index == 0 ? "" : " ") << tokens[index];
  }
  return result.str();
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

double normalized_probability_sum(const spar::Tensor& logits) {
  const spar::Tensor host{logits.to(spar::Device::cpu()).contiguous()};
  const std::size_t vocabulary{static_cast<std::size_t>(host.shape()[2])};
  const auto values{host.span<float>()};
  const float* row{values.data() + (static_cast<std::size_t>(host.shape()[1]) - 1) * vocabulary};
  const double maximum{static_cast<double>(*std::max_element(row, row + vocabulary))};
  double denominator{};
  for (std::size_t token{}; token < vocabulary; ++token) {
    denominator += std::exp(static_cast<double>(row[token]) - maximum);
  }
  double total{};
  for (std::size_t token{}; token < vocabulary; ++token) {
    total += std::exp(static_cast<double>(row[token]) - maximum) / denominator;
  }
  if (!std::isfinite(total) || std::abs(total - 1.0) > 1.0e-10) {
    throw std::runtime_error{"Next-token probability distribution failed normalization"};
  }
  return total;
}

Generation generate(const leda::Leda& model, std::span<const spar::tokenizer::TokenId> prompt,
                    const Recipe& recipe, std::uint64_t seed, const spar::Device& device) {
  leda::LedaInferenceSession session{model};
  spar::Tensor logits{session.prefill(token_tensor(prompt, device))};
  leda::Sampler sampler{{.greedy = recipe.greedy,
                         .temperature = recipe.temperature,
                         .top_k = recipe.top_k,
                         .top_p = recipe.top_p,
                         .seed = seed}};
  Generation result;
  result.tokens.reserve(maximum_generation);
  for (std::size_t step{}; step < maximum_generation; ++step) {
    const auto token{sampler.sample(logits)};
    if (token == eod_token) {
      result.eod = true;
      break;
    }
    result.tokens.push_back(token);
    if (step + 1 < maximum_generation) {
      logits = session.decode(token_tensor(std::span{&token, 1}, device));
    }
  }
  return result;
}

leda::Leda load_model(const std::filesystem::path& checkpoint, const spar::Device& device,
                      spar::MatmulPrecision precision) {
  auto loaded{spar::checkpoint::load_training_checkpoint(checkpoint)};
  leda::Leda model{leda::Leda::from_decoder(leda::leda_demo_v0(), std::move(loaded.model))};
  auto parameters{leda::parameters(model)};
  spar::nn::move_to(std::span<spar::nn::Parameter>{parameters}, device);
  spar::set_matmul_precision(device, precision);
  return model;
}

int run(int argc, char** argv) {
  if (argc != 8) {
    throw std::invalid_argument{
        "usage: leda_evaluate_generation CHECKPOINT TOKENIZER PROMPTS_TSV RECIPES_TSV "
        "OUTPUT_DIRECTORY cpu|cuda full|fp16"};
  }
  const spar::Device device{std::string_view{argv[6]} == "cpu" ? spar::Device::cpu()
                            : std::string_view{argv[6]} == "cuda"
                                ? spar::Device::cuda(0)
                                : throw std::invalid_argument{"Device must be cpu or cuda"}};
  const spar::MatmulPrecision precision{
      std::string_view{argv[7]} == "full" ? spar::MatmulPrecision::Full
      : std::string_view{argv[7]} == "fp16"
          ? spar::MatmulPrecision::Float16Inputs
          : throw std::invalid_argument{"Precision must be full or fp16"}};
  const auto tokenizer{spar::tokenizer::load_tokenizer(argv[2])};
  const auto prompt_suite{prompts(argv[3])};
  const auto recipe_grid{recipes(argv[4])};
  const std::filesystem::path output_directory{argv[5]};
  std::filesystem::create_directories(output_directory);
  std::ofstream output{output_directory / "generations.csv"};
  std::ofstream explorer{output_directory / "next-token-explorer.csv"};
  if (!output || !explorer) {
    throw std::runtime_error{"Unable to create generation-study outputs"};
  }
  output << "prompt_index,category,recipe,seed,requested_tokens,generated_tokens,eod,reached_max,"
            "token_ids,bytes_hex\n";
  explorer
      << "prompt_index,category,rank,token_id,is_eod,probability,piece_hex,full_probability_sum\n";
  leda::Leda model{load_model(argv[1], device, precision)};

  for (std::size_t prompt_index{}; prompt_index < prompt_suite.size(); ++prompt_index) {
    const Prompt& prompt{prompt_suite[prompt_index]};
    const std::vector<spar::tokenizer::TokenId> encoded{tokenizer.encode(prompt.text)};
    if (encoded.empty() || encoded.size() + maximum_generation > leda::leda_v0_max_context) {
      throw std::invalid_argument{"Scientific prompt is empty or too long for the study"};
    }
    {
      leda::LedaInferenceSession session{model};
      const spar::Tensor logits{session.prefill(token_tensor(encoded, device))};
      const double full_sum{normalized_probability_sum(logits)};
      const auto candidates{leda::top_candidates(logits, 5)};
      double previous{std::numeric_limits<double>::infinity()};
      for (std::size_t rank{}; rank < candidates.size(); ++rank) {
        const auto candidate{candidates[rank]};
        if (!std::isfinite(candidate.probability) || candidate.probability > previous) {
          throw std::runtime_error{"Top-candidate API returned invalid ordering"};
        }
        previous = candidate.probability;
        const bool is_eod{candidate.token == eod_token};
        const std::string piece{is_eod ? std::string{}
                                       : tokenizer.decode(std::span{&candidate.token, 1})};
        explorer << prompt_index << ',' << csv(prompt.category) << ',' << rank + 1 << ','
                 << candidate.token << ',' << (is_eod ? 1 : 0) << ',' << std::setprecision(17)
                 << candidate.probability << ',' << csv(hexadecimal(piece)) << ',' << full_sum
                 << '\n';
      }
    }
    for (const Recipe& recipe : recipe_grid) {
      const std::span<const std::uint64_t> seeds{recipe.greedy ? std::span{sampling_seeds}.first(1)
                                                               : std::span{sampling_seeds}};
      for (const std::uint64_t seed : seeds) {
        const Generation generated{generate(model, encoded, recipe, seed, device)};
        for (const std::size_t requested : report_lengths) {
          const std::size_t count{std::min(requested, generated.tokens.size())};
          const auto tokens{std::span{generated.tokens}.first(count)};
          const std::string bytes{tokenizer.decode(tokens)};
          const bool ended{generated.eod && generated.tokens.size() < requested};
          output << prompt_index << ',' << csv(prompt.category) << ',' << csv(recipe.name) << ','
                 << seed << ',' << requested << ',' << count << ',' << (ended ? 1 : 0) << ','
                 << (count == requested ? 1 : 0) << ',' << csv(token_list(tokens)) << ','
                 << csv(hexadecimal(bytes)) << '\n';
        }
      }
    }
    output.flush();
    explorer.flush();
    std::println("prompts={}/{}", prompt_index + 1, prompt_suite.size());
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_evaluate_generation: " << error.what() << '\n';
    return 1;
  }
}
