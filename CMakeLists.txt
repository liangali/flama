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
# FFmpeg
find_package(FFMPEG REQUIRED)
if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg: ${FFMPEG_INCLUDE_DIRS}")
endif()

# OpenVINO Runtime
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
if(OpenVINO_FOUND)
    message(STATUS "Found OpenVINO: ${OpenVINO_VERSION}")
endif()

# OpenVINO GenAI (separate package)
find_package(OpenVINOGenAI REQUIRED)
if(OpenVINOGenAI_FOUND)
    message(STATUS "Found OpenVINO GenAI: ${OpenVINOGenAI_VERSION}")
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

# Collect all source files from modular structure
file(GLOB_RECURSE FLAMA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Remove example/benchmark files from main executable (all files in examples/ folder)
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*/examples/.*")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*continuous_batching_benchmark\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*visual_language_chat\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*ffmpeg_tuning\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*vpl_infer\\.cpp$")
list(FILTER FLAMA_SOURCES EXCLUDE REGEX ".*vlm_service\\.cpp$")

# Main executable
add_executable(flama ${FLAMA_SOURCES})

# Include directories - include all src subdirectories for modular structure
target_include_directories(flama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device
    ${CMAKE_CURRENT_SOURCE_DIR}/src/video
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image
    ${CMAKE_CURRENT_SOURCE_DIR}/src/output
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch
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
    VPL::dispatcher
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/examples/continuous_batching_benchmark.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/continuous_batching_chat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/debug.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/parse_options.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/json_config.cpp
    )

    target_include_directories(flama_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
        ${OpenVINO_INCLUDE_DIRS}
    )

    target_link_libraries(flama_benchmark PRIVATE
        openvino::runtime
        openvino::genai
        nlohmann_json::nlohmann_json
    )
endif()

# Post-build: Copy runtime dependencies to output directory
if(WIN32)
    # Helper function to copy DLLs
    function(copy_dlls_to_output target_name dll_dir dll_pattern)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${target_name}>
            COMMAND ${CMAKE_COMMAND} -E echo "Copying DLLs from ${dll_dir} matching ${dll_pattern}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll_dir}/${dll_pattern}"
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endfunction()

    # 1. Copy OpenVINO Runtime DLLs
    # Try environment variable first, then fallback to common install locations
    set(OPENVINO_BIN_DIR "")
    if(DEFINED ENV{OPENVINO_DIR})
        set(OPENVINO_BIN_DIR "$ENV{OPENVINO_DIR}/../runtime/bin/intel64/Release")
    elseif(DEFINED OpenVINO_DIR)
        get_filename_component(OPENVINO_BIN_DIR "${OpenVINO_DIR}/../../../bin/intel64/Release" ABSOLUTE)
    endif()

    # Fallback: check common installation path
    if(NOT OPENVINO_BIN_DIR OR NOT EXISTS "${OPENVINO_BIN_DIR}")
        if(EXISTS "D:/library/openvino/openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64/runtime/bin/intel64/Release")
            set(OPENVINO_BIN_DIR "D:/library/openvino/openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64/runtime/bin/intel64/Release")
        endif()
    endif()

    if(OPENVINO_BIN_DIR AND EXISTS "${OPENVINO_BIN_DIR}")
        file(GLOB OPENVINO_DLLS "${OPENVINO_BIN_DIR}/*.dll")
        if(OPENVINO_DLLS)
            add_custom_command(TARGET flama POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Copying OpenVINO Runtime DLLs from ${OPENVINO_BIN_DIR}..."
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVINO_DLLS} $<TARGET_FILE_DIR:flama>
                VERBATIM
            )
        endif()

        # Copy TBB DLLs (OpenVINO dependency)
        get_filename_component(OPENVINO_ROOT "${OPENVINO_BIN_DIR}/../../../../" ABSOLUTE)
        set(TBB_BIN_DIR "${OPENVINO_ROOT}/runtime/3rdparty/tbb/bin")
        if(EXISTS "${TBB_BIN_DIR}")
            file(GLOB TBB_DLLS "${TBB_BIN_DIR}/*.dll")
            if(TBB_DLLS)
                add_custom_command(TARGET flama POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E echo "Copying TBB DLLs from ${TBB_BIN_DIR}..."
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TBB_DLLS} $<TARGET_FILE_DIR:flama>
                    VERBATIM
                )
            endif()
        endif()
    endif()

    # 2. Copy OpenVINO GenAI DLLs
    set(GENAI_BIN_DIR "")
    if(DEFINED ENV{OpenVINOGenAI_DIR})
        set(GENAI_BIN_DIR "$ENV{OpenVINOGenAI_DIR}/../runtime/bin/intel64/Release")
    elseif(DEFINED OpenVINOGenAI_DIR)
        get_filename_component(GENAI_BIN_DIR "${OpenVINOGenAI_DIR}/../../../bin/intel64/Release" ABSOLUTE)
    endif()

    # Fallback: check common installation path
    if(NOT GENAI_BIN_DIR OR NOT EXISTS "${GENAI_BIN_DIR}")
        if(EXISTS "D:/library/openvino.genai/openvino_genai_windows_2025.4.2.0_x86_64/runtime/bin/intel64/Release")
            set(GENAI_BIN_DIR "D:/library/openvino.genai/openvino_genai_windows_2025.4.2.0_x86_64/runtime/bin/intel64/Release")
        endif()
    endif()

    if(GENAI_BIN_DIR AND EXISTS "${GENAI_BIN_DIR}")
        file(GLOB GENAI_DLLS "${GENAI_BIN_DIR}/*.dll")
        if(GENAI_DLLS)
            add_custom_command(TARGET flama POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Copying OpenVINO GenAI DLLs from ${GENAI_BIN_DIR}..."
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GENAI_DLLS} $<TARGET_FILE_DIR:flama>
                VERBATIM
            )
        endif()
    endif()

    # 3. Copy VPL DLLs
    set(VPL_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/_vplinstall/bin")
    file(GLOB VPL_DLLS "${VPL_BIN_DIR}/*.dll")
    if(VPL_DLLS)
        add_custom_command(TARGET flama POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying VPL DLLs..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VPL_DLLS} $<TARGET_FILE_DIR:flama>
            VERBATIM
        )
    endif()

    # 4. Copy vcpkg DLLs (FFmpeg, etc.)
    if(DEFINED VCPKG_INSTALLED_DIR)
        set(VCPKG_BIN_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/bin")
        file(GLOB VCPKG_DLLS "${VCPKG_BIN_DIR}/*.dll")
        if(VCPKG_DLLS)
            add_custom_command(TARGET flama POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Copying vcpkg DLLs (FFmpeg, etc.)..."
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VCPKG_DLLS} $<TARGET_FILE_DIR:flama>
                VERBATIM
            )
        endif()
    elseif(EXISTS "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin")
        set(VCPKG_BIN_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin")
        file(GLOB VCPKG_DLLS "${VCPKG_BIN_DIR}/*.dll")
        if(VCPKG_DLLS)
            add_custom_command(TARGET flama POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Copying vcpkg DLLs (FFmpeg, etc.)..."
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VCPKG_DLLS} $<TARGET_FILE_DIR:flama>
                VERBATIM
            )
        endif()
    endif()

    # Print summary of DLL copy operations
    message(STATUS "")
    message(STATUS "DLL Copy Configuration:")
    if(OPENVINO_BIN_DIR AND EXISTS "${OPENVINO_BIN_DIR}")
        message(STATUS "  OpenVINO Runtime DLLs: ${OPENVINO_BIN_DIR}")
        if(TBB_BIN_DIR AND EXISTS "${TBB_BIN_DIR}")
            message(STATUS "  TBB DLLs: ${TBB_BIN_DIR}")
        endif()
    else()
        message(STATUS "  OpenVINO Runtime DLLs: NOT FOUND")
    endif()
    if(GENAI_BIN_DIR AND EXISTS "${GENAI_BIN_DIR}")
        message(STATUS "  OpenVINO GenAI DLLs: ${GENAI_BIN_DIR}")
    else()
        message(STATUS "  OpenVINO GenAI DLLs: NOT FOUND")
    endif()
    if(VPL_BIN_DIR AND EXISTS "${VPL_BIN_DIR}")
        message(STATUS "  VPL DLLs: ${VPL_BIN_DIR}")
    else()
        message(STATUS "  VPL DLLs: NOT FOUND")
    endif()
    if(DEFINED VCPKG_BIN_DIR)
        message(STATUS "  vcpkg DLLs: ${VCPKG_BIN_DIR}")
    endif()
    message(STATUS "  Target directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release")
    message(STATUS "")
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
