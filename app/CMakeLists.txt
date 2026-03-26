# CMakeLists.txt for EdgeSight Desktop Application
#
# Builds the Windows GUI application with:
#   - Qt6 (or Qt5 fallback) for UI
#   - OpenCV for webcam capture
#   - libcurl for SMS/email alerts
#   - FallDetector C++ engine for inference

cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6 first, then Qt5 as fallback
find_package(Qt6 COMPONENTS Core Gui Widgets QUIET)
if(Qt6_FOUND)
    set(QT_VERSION_MAJOR 6)
    message(STATUS "Using Qt6")
else()
    find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
    set(QT_VERSION_MAJOR 5)
    message(STATUS "Using Qt5")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Find libcurl (optional)
find_package(CURL QUIET)
if(CURL_FOUND)
    message(STATUS "libcurl found - SMS/email alerts enabled")
    set(ALERTS_ENABLED ON)
else()
    message(WARNING "libcurl not found - SMS/email alerts disabled")
    set(ALERTS_ENABLED OFF)
endif()

# Source files
set(APP_SOURCES
    main.cpp
    mainwindow.cpp
    processing_thread.cpp
    alert_manager.cpp
)

set(APP_HEADERS
    mainwindow.h
    processing_thread.h
    alert_manager.h
    config.h
)

set(APP_RESOURCES
    resources.qrc
)

# Create executable
add_executable(EdgeSightApp WIN32
    ${APP_SOURCES}
    ${APP_HEADERS}
    ${APP_RESOURCES}
)

# Set target properties
set_target_properties(EdgeSightApp PROPERTIES
    WIN32_EXECUTABLE TRUE
    OUTPUT_NAME "EdgeSight"
)

# Include directories
target_include_directories(EdgeSightApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/inference/engine/include
)

if(CURL_FOUND)
    target_include_directories(EdgeSightApp PRIVATE ${CURL_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(EdgeSightApp PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    fall_detector_lib
)

if(CURL_FOUND)
    target_link_libraries(EdgeSightApp PRIVATE ${CURL_LIBRARIES})
    target_compile_definitions(EdgeSightApp PRIVATE ALERTS_ENABLED)
endif()

# Windows-specific settings
if(WIN32)
    target_compile_definitions(EdgeSightApp PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
    
    # Windows multimedia for beep
    target_link_libraries(EdgeSightApp PRIVATE winmm)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(EdgeSightApp PRIVATE /W4 /WX-)
else()
    target_compile_options(EdgeSightApp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation
install(TARGETS EdgeSightApp
    RUNTIME DESTINATION bin
)

# Deploy Qt dependencies (Windows)
if(WIN32)
    find_program(WINDEPLOYQT windeployqt)
    if(WINDEPLOYQT)
        add_custom_command(TARGET EdgeSightApp POST_BUILD
            COMMAND ${WINDEPLOYQT} $<TARGET_FILE:EdgeSightApp>
            COMMENT "Deploying Qt dependencies..."
        )
    endif()
endif()

message(STATUS "EdgeSightApp configured successfully")
