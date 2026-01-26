cmake_minimum_required(VERSION 3.20)
project(flama VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(PkgConfig REQUIRED)

# FFmpeg
find_package(FFMPEG REQUIRED)
if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg: ${FFMPEG_INCLUDE_DIRS}")
endif()

# OpenVINO
find_package(OpenVINO REQUIRED COMPONENTS Runtime GenAI)
if(OpenVINO_FOUND)
    message(STATUS "Found OpenVINO: ${OpenVINO_VERSION}")
endif()

# nlohmann_json
find_package(nlohmann_json REQUIRED)

# oneVPL
find_package(VPL REQUIRED)
if(VPL_FOUND)
    message(STATUS "Found oneVPL")
endif()

# OpenCL
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_VERSION_STRING}")
endif()

# Collect all source files
file(GLOB_RECURSE FLAMA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Remove benchmark and test files from main executable
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*continuous_batching_benchmark\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*visual_language_chat\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*ffmpeg_tuning\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*vpl_infer\\.cpp$")

# Main executable
add_executable(flama ${FLAMA_SOURCES})

# Include directories
target_include_directories(flama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FFMPEG_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    ${VPL_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(flama PRIVATE
    ${FFMPEG_LIBRARIES}
    openvino::runtime
    openvino::genai
    nlohmann_json::nlohmann_json
    VPL::VPL
    OpenCL::OpenCL
    d3d11
    dxgi
)

# Compiler flags for Windows
if(MSVC)
    target_compile_options(flama PRIVATE
        /W4
        /MP  # Multi-processor compilation
        /permissive-
        /Zc:__cplusplus
        /EHsc
    )
    target_compile_definitions(flama PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
else()
    target_compile_options(flama PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Optional: Build benchmark executable
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)
if(BUILD_BENCHMARKS)
    add_executable(flama_benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/src/continuous_batching_benchmark.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/continuous_batching_chat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/parse_options.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/json_config.cpp
    )

    target_include_directories(flama_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenVINO_INCLUDE_DIRS}
    )

    target_link_libraries(flama_benchmark PRIVATE
        openvino::runtime
        openvino::genai
        nlohmann_json::nlohmann_json
    )
endif()

# Install targets
install(TARGETS flama
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install config examples
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config/
    DESTINATION share/flama/config
    FILES_MATCHING PATTERN "*.json"
    PATTERN "*.example"
)

# Print configuration summary
message(STATUS "")
message(STATUS "Flama Configuration Summary:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "")
