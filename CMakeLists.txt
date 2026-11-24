cmake_minimum_required(VERSION 3.20)
project(TriggerDAQ VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TRIGGERDAQ_BUILD_TESTS "Build C++ unit tests" ON)
option(TRIGGERDAQ_BUILD_BENCHMARKS "Build C++ benchmarks" ON)
option(TRIGGERDAQ_BUILD_PYTHON "Build nanobind Python extension" OFF)
option(TRIGGERDAQ_ENABLE_NATIVE_SIMD "Enable AVX2 and SSE4.2 SIMD flags when supported" ON)

find_package(Threads REQUIRED)

# Compiler flags
set(TRIGGERDAQ_COMPILE_OPTIONS
    -O3
    -Wall
    -Wextra
    -Werror
    -Wno-missing-field-initializers
)

include(CheckCXXCompilerFlag)
if(TRIGGERDAQ_ENABLE_NATIVE_SIMD)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    check_cxx_compiler_flag("-mcx16" COMPILER_SUPPORTS_CX16)
    if(COMPILER_SUPPORTS_AVX2)
        list(APPEND TRIGGERDAQ_COMPILE_OPTIONS -mavx2)
    endif()
    if(COMPILER_SUPPORTS_SSE42)
        list(APPEND TRIGGERDAQ_COMPILE_OPTIONS -msse4.2)
    endif()
    if(COMPILER_SUPPORTS_CX16)
        list(APPEND TRIGGERDAQ_COMPILE_OPTIONS -mcx16)
    endif()
endif()

# Core static library
add_library(triggerdaq_core STATIC
    src/core/slab_arena.cpp
    src/core/telemetry.cpp
    src/frontend/detector_emulator.cpp
    src/eb/event_builder.cpp
    src/l1/simd_kernels.cpp
    src/l1/prescaler.cpp
    src/l1/l1_trigger.cpp
    src/hlt/reconstruction.cpp
    src/hlt/classifier.cpp
    src/hlt/hlt_farm.cpp
    src/storage/event_writer.cpp
    src/engine.cpp
)

set_target_properties(triggerdaq_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(triggerdaq_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(triggerdaq_core PRIVATE ${TRIGGERDAQ_COMPILE_OPTIONS})
target_link_libraries(triggerdaq_core PUBLIC Threads::Threads)

# C++ Tests
if(TRIGGERDAQ_BUILD_TESTS)
    enable_testing()

    function(triggerdaq_add_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE triggerdaq_core)
        target_compile_options(${name} PRIVATE ${TRIGGERDAQ_COMPILE_OPTIONS})
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    triggerdaq_add_test(test_slab_arena tests/cpp/test_slab_arena.cpp)
    triggerdaq_add_test(test_l1_simd tests/cpp/test_l1_simd.cpp)
    triggerdaq_add_test(test_prescaler tests/cpp/test_prescaler.cpp)
    triggerdaq_add_test(test_storage_crc tests/cpp/test_storage_crc.cpp)
    triggerdaq_add_test(test_shutdown_drain tests/cpp/test_shutdown_drain.cpp)
    triggerdaq_add_test(test_cern_upgrades tests/cpp/test_cern_upgrades.cpp)
endif()

# C++ Benchmarks
if(TRIGGERDAQ_BUILD_BENCHMARKS)
    add_executable(bench_pipeline benchmarks/bench_pipeline.cpp)
    target_link_libraries(bench_pipeline PRIVATE triggerdaq_core)
    target_compile_options(bench_pipeline PRIVATE ${TRIGGERDAQ_COMPILE_OPTIONS})
endif()

# Python nanobind extension
if(TRIGGERDAQ_BUILD_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
        RESULT_VARIABLE nanobind_RES
    )
    if(NOT nanobind_RES EQUAL 0)
        message(FATAL_ERROR "Failed to locate nanobind cmake dir via python -m nanobind --cmake_dir")
    endif()
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_triggerdaq_ext
        src/bindings/py_module.cpp
    )
    target_link_libraries(_triggerdaq_ext PRIVATE triggerdaq_core)
    target_compile_options(_triggerdaq_ext PRIVATE ${TRIGGERDAQ_COMPILE_OPTIONS})

    install(TARGETS _triggerdaq_ext LIBRARY DESTINATION triggerdaq)
endif()
