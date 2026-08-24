import spar;
import std;

namespace {

[[noreturn]] void usage() {
  throw std::invalid_argument{"usage: leda_tokenizer_compare OLD_TOKENIZER "
                              "NEW_TOKENIZER PATH_MANIFEST... | "
                              "OLD_TOKENIZER NEW_TOKENIZER --labeled-tsv TSV SPLIT"};
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
  std::ifstream stream{path, std::ios::binary};
  if (!stream) {
    throw std::runtime_error{"Unable to open document: " + path.string()};
  }
  return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

struct Counts final {
  std::uint64_t documents{};
  std::uint64_t bytes{};
  std::uint64_t old_tokens{};
  std::uint64_t new_tokens{};
};

void add(Counts& result, std::size_t bytes, std::size_t old_tokens, std::size_t new_tokens) {
  ++result.documents;
  result.bytes += bytes;
  result.old_tokens += old_tokens;
  result.new_tokens += new_tokens;
}

void print_counts(std::string_view view, const Counts& counts) {
  if (counts.old_tokens == 0 || counts.new_tokens == 0) {
    throw std::runtime_error{"Tokenizer comparison produced no tokens"};
  }
  std::println("{},{},{},{},{},{:.8f},{:.8f},{:.8f}", view, counts.documents, counts.bytes,
               counts.old_tokens, counts.new_tokens,
               static_cast<double>(counts.bytes) / static_cast<double>(counts.old_tokens),
               static_cast<double>(counts.bytes) / static_cast<double>(counts.new_tokens),
               static_cast<double>(counts.new_tokens) / static_cast<double>(counts.old_tokens));
}

Counts measure(const spar::tokenizer::ByteBPETokenizer& old_tokenizer,
               const spar::tokenizer::ByteBPETokenizer& new_tokenizer,
               const std::filesystem::path& manifest) {
  Counts result;
  for (const auto& path : read_paths(manifest)) {
    const std::string document{read_document(path)};
    add(result, document.size(), old_tokenizer.encode(document).size(),
        new_tokenizer.encode(document).size());
  }
  return result;
}

std::vector<std::string_view> fields(std::string_view line) {
  std::vector<std::string_view> result;
  while (true) {
    const std::size_t tab{line.find('\t')};
    result.push_back(line.substr(0, tab));
    if (tab == std::string_view::npos) {
      return result;
    }
    line.remove_prefix(tab + 1);
  }
}

void measure_labeled(const spar::tokenizer::ByteBPETokenizer& old_tokenizer,
                     const spar::tokenizer::ByteBPETokenizer& new_tokenizer,
                     const std::filesystem::path& manifest, std::string_view selected_split) {
  std::ifstream stream{manifest};
  if (!stream) {
    throw std::runtime_error{"Unable to open labeled manifest: " + manifest.string()};
  }
  std::string line;
  if (!std::getline(stream, line) ||
      line != "split\tsource_family\tbroad_domain\tlicense\tlocal_document_path") {
    throw std::runtime_error{"Invalid labeled-manifest header: " + manifest.string()};
  }
  Counts overall;
  std::map<std::string, Counts> sources;
  std::map<std::string, Counts> domains;
  std::map<std::string, Counts> licenses;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto values{fields(line)};
    if (values.size() != 5) {
      throw std::runtime_error{"Invalid labeled-manifest row: " + manifest.string()};
    }
    if (values[0] != selected_split) {
      continue;
    }
    std::filesystem::path path{values[4]};
    if (path.is_relative()) {
      path = manifest.parent_path() / path;
    }
    const std::string document{read_document(path)};
    const std::size_t old_tokens{old_tokenizer.encode(document).size()};
    const std::size_t new_tokens{new_tokenizer.encode(document).size()};
    add(overall, document.size(), old_tokens, new_tokens);
    add(sources[std::string{values[1]}], document.size(), old_tokens, new_tokens);
    add(domains[std::string{values[2]}], document.size(), old_tokens, new_tokens);
    add(licenses[std::string{values[3]}], document.size(), old_tokens, new_tokens);
  }
  if (!stream.eof()) {
    throw std::runtime_error{"Labeled manifest is unreadable: " + manifest.string()};
  }
  print_counts(std::string{selected_split} + "-overall", overall);
  for (const auto& [label, counts] : sources) {
    print_counts(std::string{selected_split} + "-source-" + label, counts);
  }
  for (const auto& [label, counts] : domains) {
    print_counts(std::string{selected_split} + "-domain-" + label, counts);
  }
  for (const auto& [label, counts] : licenses) {
    print_counts(std::string{selected_split} + "-license-" + label, counts);
  }
}

void inspect(const spar::tokenizer::ByteBPETokenizer& old_tokenizer,
             const spar::tokenizer::ByteBPETokenizer& new_tokenizer) {
  for (const std::string_view text :
       {"interleukin-6", "CD19", "β-catenin", "N-acetylcysteine", "TP53", "phosphorylation",
        "double-stranded DNA", "mg/kg", "p < 0.05", "∇·E = ρ/ε₀"}) {
    const auto old_tokens{old_tokenizer.encode(text)};
    const auto new_tokens{new_tokenizer.encode(text)};
    std::cerr << "inspect " << std::quoted(text) << " old_tokens=" << old_tokens.size()
              << " new_tokens=" << new_tokens.size() << '\n';
  }
}

int run(int argc, char** argv) {
  if (argc < 4) {
    usage();
  }
  const auto old_tokenizer{spar::tokenizer::load_tokenizer(argv[1])};
  const auto new_tokenizer{spar::tokenizer::load_tokenizer(argv[2])};
  std::println("view,documents,bytes,old_tokens,new_tokens,old_bytes_per_token,"
               "new_bytes_per_token,new_vs_old_ratio");
  if (argc == 6 && std::string_view{argv[3]} == "--labeled-tsv") {
    measure_labeled(old_tokenizer, new_tokenizer, argv[4], argv[5]);
    inspect(old_tokenizer, new_tokenizer);
    return 0;
  }
  for (int index{3}; index < argc; ++index) {
    const std::filesystem::path manifest{argv[index]};
    const Counts counts{measure(old_tokenizer, new_tokenizer, manifest)};
    print_counts(manifest.stem().string(), counts);
  }
  inspect(old_tokenizer, new_tokenizer);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "leda_tokenizer_compare: " << error.what() << '\n';
    return 1;
  }
}
