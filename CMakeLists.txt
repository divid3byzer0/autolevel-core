cmake_minimum_required(VERSION 3.16)
project(autolevel_core VERSION 1.0.0 LANGUAGES CXX)

# Header-only. Consumers (autolevel-box, autolevel-dj-vst) add this directory with
# add_subdirectory() and link the `autolevel::core` target.
add_library(autolevel_core INTERFACE)
add_library(autolevel::core ALIAS autolevel_core)
target_include_directories(autolevel_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
)
target_compile_features(autolevel_core INTERFACE cxx_std_17)

# Tests are on when this repo is built on its own, off when it is a submodule.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_autolevel_core_top ON)
else()
    set(_autolevel_core_top OFF)
endif()
option(AUTOLEVEL_CORE_BUILD_TESTS "Build the autolevel-core DSP tests" ${_autolevel_core_top})

if (AUTOLEVEL_CORE_BUILD_TESTS)
    find_package(Threads REQUIRED)
    add_executable(autolevel_core_dsp_test tests/dsp_test.cpp)
    target_link_libraries(autolevel_core_dsp_test PRIVATE autolevel::core Threads::Threads)
    if (MSVC)
        target_compile_options(autolevel_core_dsp_test PRIVATE /W4 /O2 /fp:fast)
    else()
        # Same flags the box ships with.
        target_compile_options(autolevel_core_dsp_test PRIVATE
            -Wall -Wextra -O3 -ffast-math -fno-finite-math-only)
    endif()
    # The suite is built on assert(). Release defines NDEBUG, which would compile every
    # check out and still print "ALL DSP TESTS PASSED". Strip it for this target only.
    target_compile_options(autolevel_core_dsp_test PRIVATE -UNDEBUG)
endif()
