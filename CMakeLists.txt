cmake_minimum_required(VERSION 3.12)
project("berts.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#
# Option list
#

# general
option(BERTS_STATIC                 "bert: static link libraries"                          OFF)
option(BERTS_NATIVE                 "bert: enable -march=native flag"                       OFF)
option(BERTS_LTO                    "bert: enable link time optimization"                  OFF)

# debug
option(BERTS_ALL_WARNINGS           "bert: enable all compiler warnings"                   OFF)
option(BERTS_ALL_WARNINGS_3RD_PARTY "bert: enable all compiler warnings in 3rd party libs" OFF)
option(BERTS_GPROF                  "bert: enable gprof"                                   OFF)
option(BERTS_PERF                   "bert: enable performance logging"                     OFF)

# sanitizers
option(BERTS_SANITIZE_THREAD        "bert: enable thread sanitizer"                        OFF)
option(BERTS_SANITIZE_ADDRESS       "bert: enable address sanitizer"                       OFF)
option(BERTS_SANITIZE_UNDEFINED     "bert: enable undefined sanitizer"                     OFF)

# instruction set specific
option(BERTS_AVX                    "bert: enable AVX"                                     ON)
option(BERTS_AVX2                   "bert: enable AVX2"                                    ON)
option(BERTS_FMA                    "bert: enable FMA"                                     ON)

# 3rd party libs
option(BERTS_ACCELERATE             "bert: enable Accelerate framework"                    ON)
option(BERTS_OPENBLAS               "bert: use OpenBLAS"                                   OFF)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (BERTS_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (BERTS_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (BERTS_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND BERTS_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(BERTS_EXTRA_LIBS ${BERTS_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()
if (BERTS_OPENBLAS)
    if (BERTS_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (BERTS_ALL_WARNINGS)
    if (NOT MSVC)
        add_compile_options(
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
        )
    else()
        # todo : msvc
    endif()
endif()

if(NOT WIN32)
add_compile_options(-Wno-format)
endif()

if(BERTS_PERF)
    add_compile_definitions(GGML_PERF)
endif()

if (NOT MSVC)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
else()
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Ox")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox")
endif()

if (BERTS_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (BERTS_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (BERTS_GPROF)
        add_compile_options(-pg)
    endif()
    if (BERTS_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (NOT MSVC)
        add_compile_options(-mf16c)
        if (BERTS_FMA)
            add_compile_options(-mfma)
        endif()
        if (BERTS_AVX)
            add_compile_options(-mavx)
        endif()
        if (BERTS_AVX2)
            add_compile_options(-mavx2)
        endif()
    endif()
else()
    # TODO: support PowerPC
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(ggml)

add_library(tokenization
            tokenization.cpp
            tokenization.h)

add_library(bertbase
            bertbase.cpp
            bertbase.h)

add_library(distilbert
            distilbert.cpp
            distilbert.h)

add_library(bertencoder
            bertencoder.cpp
            bertencoder.h)

target_include_directories(bertbase PUBLIC .)
target_compile_features(bertbase PUBLIC cxx_std_20)
target_link_libraries(bertbase PRIVATE tokenization ggml ${BERTS_EXTRA_LIBS})

target_include_directories(distilbert PUBLIC .)
target_compile_features(distilbert PUBLIC cxx_std_20)
target_link_libraries(distilbert PRIVATE tokenization ggml bertbase ${BERTS_EXTRA_LIBS})

target_include_directories(bertencoder PUBLIC .)
target_compile_features(bertencoder PUBLIC cxx_std_20)
target_link_libraries(bertencoder PRIVATE tokenization ggml bertbase ${BERTS_EXTRA_LIBS})

if(WIN32)
include_directories(${CMAKE_SOURCE_DIR}/include_win)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(tokenization PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(distilbert PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(bertencoder PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(bertbase PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(distilbert PRIVATE DISTILBERTS_SHARED DISTILBERTS_BUILD)
    target_compile_definitions(bertencoder PRIVATE DISTILBERTS_SHARED DISTILBERTS_BUILD)
    target_compile_definitions(bertbase PRIVATE DISTILBERTS_SHARED DISTILBERTS_BUILD)
else()
    add_subdirectory(examples)
endif()