# Root CMakeLists.txt for EdgeSight Project
#
# Builds:
#   - C++ inference engine (inference/engine/)
#   - Windows desktop application (app/)
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   cmake --build . --config Release

cmake_minimum_required(VERSION 3.20)

# Project
project(EdgeSight VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(BUILD_APP "Build Windows desktop application" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# Compiler settings
if(MSVC)
    # MSVC settings
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    # Disable specific warnings that are too noisy
    add_compile_options(/wd4251 /wd4275)
else()
    # GCC/Clang settings
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-fPIC)
endif()

# Find packages
find_package(Threads REQUIRED)

# OpenCV (for app)
if(BUILD_APP)
    find_package(OpenCV QUIET)
    if(NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Desktop app will be disabled.")
        set(BUILD_APP OFF)
    endif()
endif()

# Qt6 (for app GUI)
if(BUILD_APP)
    find_package(Qt6 COMPONENTS Core Gui Widgets QUIET)
    if(NOT Qt6_FOUND)
        find_package(Qt5 COMPONENTS Core Gui Widgets QUIET)
        if(NOT Qt5_FOUND)
            message(WARNING "Qt not found. Desktop app will be disabled.")
            set(BUILD_APP OFF)
        endif()
    endif()
endif()

# libcurl (for alerts)
if(BUILD_APP)
    find_package(CURL QUIET)
    if(NOT CURL_FOUND)
        message(WARNING "libcurl not found. SMS/email alerts will be disabled.")
    endif()
endif()

# Add subdirectories
add_subdirectory(inference/engine)

if(BUILD_APP)
    add_subdirectory(app)
endif()

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "EdgeSight")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VENDOR "EdgeSight Project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Real-time fall detection using AI")
include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "EdgeSight Build Configuration")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  Inference Engine: YES")
message(STATUS "  Unit Tests: ${BUILD_TESTS}")
message(STATUS "  Desktop App: ${BUILD_APP}")
if(BUILD_APP)
    message(STATUS "    OpenCV: ${OpenCV_VERSION}")
    if(Qt6_FOUND)
        message(STATUS "    Qt: 6.x")
    elseif(Qt5_FOUND)
        message(STATUS "    Qt: 5.x")
    endif()
    message(STATUS "    libcurl: ${CURL_FOUND}")
endif()
message(STATUS "")
message(STATUS "Output directories:")
message(STATUS "  Binaries: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  Libraries: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "========================================")
message(STATUS "")
