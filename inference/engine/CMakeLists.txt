# CMakeLists.txt for EdgeSight C++ Inference Engine
#
# Builds:
#   - fall_detector_lib: Static library for inference
#   - test_engine: Google Test executable

cmake_minimum_required(VERSION 3.20)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project
project(EdgeSightEngine VERSION 1.0.0 LANGUAGES CXX)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(USE_SYSTEM_ONNXRUNTIME "Use system ONNX Runtime" OFF)

# Find dependencies
find_package(Threads REQUIRED)

# ONNX Runtime
if(USE_SYSTEM_ONNXRUNTIME)
    # Use system-installed ONNX Runtime
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS /usr/local/include/onnxruntime
              ${CMAKE_PREFIX_PATH}/include/onnxruntime
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS /usr/local/lib
              ${CMAKE_PREFIX_PATH}/lib
    )
    
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "System ONNX Runtime not found. Set USE_SYSTEM_ONNXRUNTIME=OFF to download.")
    endif()
    
    message(STATUS "Using system ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
    # Download ONNX Runtime
    set(ONNXRUNTIME_VERSION "1.18.0")
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
    
    if(WIN32)
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
        set(ONNXRUNTIME_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.zip")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.tgz")
    endif()
    
    if(NOT EXISTS ${ONNXRUNTIME_ROOT})
        message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")
        file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE} SHOW_PROGRESS)
        
        if(WIN32)
            execute_process(
                COMMAND powershell -Command "Expand-Archive -Path ${ONNXRUNTIME_ARCHIVE} -DestinationPath ${CMAKE_CURRENT_SOURCE_DIR}/third_party -Force"
            )
            file(RENAME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}" ${ONNXRUNTIME_ROOT})
        else()
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party
            )
            file(RENAME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" ${ONNXRUNTIME_ROOT})
        endif()
        
        file(REMOVE ${ONNXRUNTIME_ARCHIVE})
    endif()
    
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
    
    if(WIN32)
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    else()
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
    endif()
    
    message(STATUS "Using downloaded ONNX Runtime: ${ONNXRUNTIME_ROOT}")
endif()

# Source files
set(ENGINE_SOURCES
    src/fall_detector.cpp
)

set(ENGINE_HEADERS
    include/fall_detector.h
)

# Static library
add_library(fall_detector_lib STATIC ${ENGINE_SOURCES} ${ENGINE_HEADERS})

target_include_directories(fall_detector_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(fall_detector_lib
    PUBLIC
        Threads::Threads
        ${ONNXRUNTIME_LIB}
)

# Compiler warnings (treat as errors for production)
if(MSVC)
    target_compile_options(fall_detector_lib PRIVATE /W4 /WX-)
else()
    target_compile_options(fall_detector_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Tests
if(BUILD_TESTS)
    # Fetch Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    enable_testing()
    
    add_executable(test_engine
        tests/test_engine.cpp
    )
    
    target_link_libraries(test_engine
        PRIVATE
            fall_detector_lib
            GTest::gtest
            GTest::gtest_main
    )
    
    if(MSVC)
        target_compile_options(test_engine PRIVATE /W4 /WX-)
    else()
        target_compile_options(test_engine PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    
    # Copy ONNX Runtime DLL to build directory (Windows)
    if(WIN32 AND EXISTS ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll)
        add_custom_command(TARGET test_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
                $<TARGET_FILE_DIR:test_engine>
        )
    endif()
    
    include(GoogleTest)
    gtest_discover_tests(test_engine)
    
    # Build C++ benchmark binary for Python comparison
    add_executable(cpp_benchmark
        ../benchmark/cpp_benchmark.cpp
    )
    
    target_link_libraries(cpp_benchmark
        PRIVATE
            fall_detector_lib
    )
    
    if(MSVC)
        target_compile_options(cpp_benchmark PRIVATE /W4 /WX-)
    else()
        target_compile_options(cpp_benchmark PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    
    # Copy ONNX Runtime DLL for benchmark
    if(WIN32 AND EXISTS ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll)
        add_custom_command(TARGET cpp_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
                $<TARGET_FILE_DIR:cpp_benchmark>
        )
    endif()
endif()

# Installation
install(TARGETS fall_detector_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${ENGINE_HEADERS}
    DESTINATION include
)

# Print summary
message(STATUS "")
message(STATUS "EdgeSight Inference Engine Configuration:")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
