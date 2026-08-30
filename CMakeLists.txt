cmake_minimum_required(VERSION 4.4)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

if(APPLE)
    if(NOT CMAKE_OSX_SYSROOT)
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            RESULT_VARIABLE LEDA_XCRUN_RESULT
            OUTPUT_VARIABLE LEDA_MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(LEDA_XCRUN_RESULT EQUAL 0)
            set(CMAKE_OSX_SYSROOT "${LEDA_MACOS_SDK_PATH}" CACHE PATH "macOS SDK path")
        endif()
    endif()

    if(NOT DEFINED CMAKE_CXX_STDLIB_MODULES_JSON)
        foreach(LEDA_LLVM_PREFIX IN ITEMS /opt/homebrew/opt/llvm /usr/local/opt/llvm)
            if(EXISTS "${LEDA_LLVM_PREFIX}/lib/c++/libc++.modules.json")
                set(
                    CMAKE_CXX_STDLIB_MODULES_JSON
                    "${LEDA_LLVM_PREFIX}/lib/c++/libc++.modules.json"
                    CACHE FILEPATH "C++ standard-library module manifest"
                )
                break()
            endif()
        endforeach()
    endif()
endif()

project(
    Leda
    VERSION 0.1.0
    DESCRIPTION "A science and biomedical language-model project built with Spar"
    LANGUAGES CXX
)

set(CMAKE_CXX_MODULE_STD ON)

option(LEDA_BUILD_TESTS "Build Leda tests" OFF)
option(LEDA_BUILD_EXAMPLES "Build Leda examples" OFF)
option(LEDA_BUILD_TOOLS "Build Leda offline training tools" OFF)
set(SPAR_SOURCE_DIR "" CACHE PATH "Path to a Spar source checkout")

if(NOT CMAKE_GENERATOR MATCHES "^Ninja")
    message(FATAL_ERROR
        "Leda's C++23 module build currently requires Ninja or Ninja Multi-Config. "
        "Detected generator: ${CMAKE_GENERATOR}.")
endif()

if(NOT 23 IN_LIST CMAKE_CXX_COMPILER_IMPORT_STD
   OR CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE)
    message(FATAL_ERROR
        "Leda requires C++23 'import std' support. Use a recent LLVM Clang with "
        "libc++ standard-library modules and Ninja.")
endif()

if(NOT TARGET Spar::Spar)
    if(SPAR_SOURCE_DIR)
        get_filename_component(LEDA_SPAR_SOURCE_DIR "${SPAR_SOURCE_DIR}" ABSOLUTE
                               BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${LEDA_SPAR_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "SPAR_SOURCE_DIR does not contain a Spar CMakeLists.txt: "
                "${LEDA_SPAR_SOURCE_DIR}")
        endif()
        block()
            set(SPAR_BUILD_TESTS OFF)
            set(SPAR_BUILD_EXAMPLES OFF)
            set(SPAR_BUILD_TOOLS OFF)
            add_subdirectory("${LEDA_SPAR_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/spar-build"
                             EXCLUDE_FROM_ALL)
        endblock()
    else()
        find_package(Spar CONFIG REQUIRED)
    endif()
endif()

add_library(leda)
add_library(Leda::Leda ALIAS leda)

target_compile_features(leda PUBLIC cxx_std_23)
set_target_properties(leda PROPERTIES CXX_EXTENSIONS OFF EXPORT_NAME Leda)
target_link_libraries(leda PUBLIC Spar::Spar)

target_sources(leda
    PRIVATE
        src/config.cpp
        src/inference.cpp
        src/model.cpp
        src/pretraining.cpp
    PUBLIC
        FILE_SET leda_modules TYPE CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
        FILES
            src/leda.cppm
            src/config.cppm
            src/inference.cppm
            src/model.cppm
            src/pretraining.cppm
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(leda PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
elseif(MSVC)
    target_compile_options(leda PRIVATE /W4 /permissive-)
endif()

if(LEDA_BUILD_TESTS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        message(FATAL_ERROR
            "LEDA_BUILD_TESTS was enabled, but the local Leda test tree is not present. "
            "The public repository does not currently ship the development test suite.")
    endif()
    enable_testing()
    add_subdirectory(tests)
endif()

if(LEDA_BUILD_EXAMPLES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
        message(FATAL_ERROR
            "LEDA_BUILD_EXAMPLES was enabled, but the local Leda example tree is not present. "
            "The public repository does not currently ship the development examples.")
    endif()
    add_subdirectory(examples)
endif()

if(LEDA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
