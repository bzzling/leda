add_executable(leda_train_tokenizer train_tokenizer.cpp)
add_executable(leda_tokenizer_compare tokenizer_compare.cpp)
add_executable(leda_evaluate_documents evaluate_documents.cpp)
add_executable(leda_evaluate_prefixes evaluate_prefixes.cpp)
add_executable(leda_pretrain_sharded pretrain_sharded.cpp)
add_executable(leda_model_info model_info.cpp)
add_executable(leda_complete complete.cpp)
add_executable(leda_inference_benchmark inference_benchmark.cpp)
add_executable(leda_evaluate_autocomplete evaluate_autocomplete.cpp)
add_executable(leda_evaluate_generation evaluate_generation.cpp)

foreach(tool IN ITEMS leda_train_tokenizer leda_tokenizer_compare leda_evaluate_documents
                      leda_evaluate_prefixes leda_complete leda_inference_benchmark
                      leda_evaluate_autocomplete leda_evaluate_generation
                      leda_pretrain_sharded leda_model_info)
    target_compile_features(${tool} PRIVATE cxx_std_23)
    set_target_properties(${tool} PROPERTIES CXX_EXTENSIONS OFF)
endforeach()
target_link_libraries(leda_train_tokenizer PRIVATE Spar::Spar)
target_link_libraries(leda_tokenizer_compare PRIVATE Spar::Spar)
target_link_libraries(leda_evaluate_documents PRIVATE Leda::Leda)
target_link_libraries(leda_evaluate_prefixes PRIVATE Leda::Leda)
target_link_libraries(leda_pretrain_sharded PRIVATE Leda::Leda)
target_link_libraries(leda_model_info PRIVATE Leda::Leda)
target_link_libraries(leda_complete PRIVATE Leda::Leda)
target_link_libraries(leda_inference_benchmark PRIVATE Leda::Leda)
target_link_libraries(leda_evaluate_autocomplete PRIVATE Leda::Leda)
target_link_libraries(leda_evaluate_generation PRIVATE Leda::Leda)

if(SPAR_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(leda_inference_benchmark PRIVATE LEDA_BENCHMARK_CUDA=1)
    target_link_libraries(leda_inference_benchmark PRIVATE CUDA::cudart)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(tool IN ITEMS leda_train_tokenizer leda_tokenizer_compare leda_evaluate_documents
                          leda_evaluate_prefixes leda_complete leda_inference_benchmark
                          leda_evaluate_autocomplete leda_evaluate_generation
                          leda_pretrain_sharded leda_model_info)
        target_compile_options(${tool} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endforeach()
elseif(MSVC)
    foreach(tool IN ITEMS leda_train_tokenizer leda_tokenizer_compare leda_evaluate_documents
                          leda_evaluate_prefixes leda_complete leda_inference_benchmark
                          leda_evaluate_autocomplete leda_evaluate_generation
                          leda_pretrain_sharded leda_model_info)
        target_compile_options(${tool} PRIVATE /W4 /permissive-)
    endforeach()
endif()
