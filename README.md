# Flama - Hardware-Accelerated Video Processing Pipeline

Flama is a C++ video processing pipeline that demonstrates advanced hardware-accelerated video decoding, processing, and AI-driven scene understanding using FFmpeg, Intel OpenVINO, oneVPL, and OpenCL.

## Features

- **Hardware-Accelerated Video Decoding**: GPU-based decoding using FFmpeg with D3D11VA
- **Intelligent Frame Selection**: Multiple sampling strategies (frame interval, time window, keyframe priority)
- **Video Post-Processing (VPP)**: Hardware-accelerated scaling and format conversion via oneVPL
- **Scene Understanding**: OpenVINO GenAI Visual Language Model (VLM) integration
- **Continuous Batching**: Efficient asynchronous batch inference
- **Performance Profiling**: Detailed frame-level and batch-level timing statistics
- **Zero-Copy GPU Pipeline**: Minimizes CPU-GPU memory transfers

## Prerequisites

- **Operating System**: Windows 10+ with D3D11 support
- **Compiler**: Visual Studio 2022 (or Visual Studio 17 2022) with C++17 support
- **CMake**: Version 3.20 or higher
- **vcpkg**: For dependency management
- **GPU**: Intel GPU recommended for optimal performance (supports D3D11VA)

## Dependencies

The following dependencies are managed via vcpkg:

- **FFmpeg 6.0+**: Video decoding and format handling
- **nlohmann-json 3.11+**: JSON configuration parsing
- **OpenCL 3.0+**: GPU computing (optional)

The following dependencies need to be installed manually:

- **Intel OpenVINO Runtime 2025.4+**: AI inference engine ([Download](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html))
- **OpenVINO GenAI 2025.4+**: Visual Language Model support ([Download](https://github.com/openvinotoolkit/openvino.genai))
- **Intel oneVPL**: Video post-processing ([Download](https://github.com/oneapi-src/oneVPL))

## Environment Setup

### 1. Install vcpkg

If you haven't installed vcpkg yet:

```powershell
# Clone vcpkg
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Add vcpkg to PATH (optional but recommended)
# Add C:\vcpkg to your system PATH environment variable
```

Set the vcpkg environment variable:

```powershell
# Set VCPKG_ROOT environment variable
setx VCPKG_ROOT "C:\vcpkg"
```

### 2. Install OpenVINO

Download and install Intel OpenVINO Runtime and OpenVINO GenAI:

1. Visit [OpenVINO Downloads](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
2. Download the Windows installer
3. Install to default location (e.g., `C:\Program Files (x86)\Intel\openvino_2025`)
4. Set environment variables:

```powershell
# Add to system PATH
setx PATH "%PATH%;C:\Program Files (x86)\Intel\openvino_2025\runtime\bin\intel64\Release"

# Set OpenVINO environment
setx OPENVINO_DIR "C:\Program Files (x86)\Intel\openvino_2025"
```

Download OpenVINO GenAI:

```powershell
# Download pre-built binaries from GitHub releases
# Or build from source: https://github.com/openvinotoolkit/openvino.genai

# Extract to a location, e.g., C:\openvino.genai
# Set environment variable
setx OPENVINO_GENAI_DIR "C:\openvino.genai"
```

### 3. Install oneVPL

Download and install Intel oneVPL:

1. Visit [oneVPL GitHub Releases](https://github.com/oneapi-src/oneVPL/releases)
2. Download the Windows installer or build from source
3. Install to default location (e.g., `C:\Program Files\Intel\oneVPL`)
4. Set environment variables:

```powershell
setx VPL_DIR "C:\Program Files\Intel\oneVPL"
setx PATH "%PATH%;C:\Program Files\Intel\oneVPL\bin"
```

### 4. Install Dependencies via vcpkg

Navigate to the project directory and install dependencies:

```powershell
cd D:\code\flama_code\flama

# Install dependencies using vcpkg manifest mode
vcpkg install --triplet=x64-windows
```

This will install:
- FFmpeg (with required features)
- nlohmann-json
- OpenCL

**Note**: vcpkg will automatically integrate with CMake when you configure the build.

## Building the Project

### Option 1: Using CMake with vcpkg Integration

```powershell
# Navigate to project directory
cd D:\code\flama_code\flama

# Create and enter build directory
mkdir build
cd build

# Configure CMake with vcpkg toolchain
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --config Release

# The executable will be in: build\bin\Release\flama.exe
```

### Option 2: Using vcpkg Manifest Mode (Recommended)

```powershell
# Navigate to project directory
cd D:\code\flama_code\flama

# Create build directory
mkdir build
cd build

# Configure with vcpkg automatic integration
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

# Build
cmake --build . --config Release

# The executable will be in: build\bin\Release\flama.exe
```

### Build Options

You can enable optional features during CMake configuration:

```powershell
# Build with benchmark executables
cmake .. -DBUILD_BENCHMARKS=ON `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
```

## Running the Application

### Basic Usage

```powershell
# Navigate to build output
cd D:\code\flama_code\flama\build\bin\Release

# Run with hardware decode
.\flama.exe "path\to\video.mp4" hw

# Run with software decode
.\flama.exe "path\to\video.mp4" sw
```

### Command-Line Options

```powershell
# Basic syntax
flama.exe <input_video> <hw|sw> [output_dir] [options]

# Frame selection options
--sel-policy=<policy>        # frame, time, mixed, key, mixed-key
--frame-interval=60          # Sample every N frames
--window-seconds=1.0         # Time window in seconds
--max-per-window=2           # Max frames per window
--min-frames-between=0       # Min frame interval
--no-force-keyframe          # Don't force keyframe selection

# Batch processing options
--batch-trigger=10           # Batch size trigger
--max-cache=128              # Max cache size
--new-batch-mode             # Use new batch mode

# Continuous batching options
--use_cb                     # Enable continuous batching
--cb_batch_size=10           # CB batch size
--cb-multi-thread            # Enable CB multi-threading
--new-multithread            # Use new multi-thread mode

# Scheduler options
--max_num_seqs=64            # Max sequences
--dynamic_split_fuse=false   # Dynamic split/fuse

# Debug options
--debug, -d                  # Enable debug mode
```

### Using JSON Configuration

Create a configuration file based on `config/example_config.json`:

```powershell
# Copy example config
copy ..\config\example_config.json ..\config\my_config.json

# Edit my_config.json with your settings

# Run with config file
.\flama.exe "path\to\video.mp4" hw --config=..\config\my_config.json
```

## Usage Examples

### Example 1: Hardware Decode with Frame Interval

```powershell
.\flama.exe "D:\videos\sample.mp4" hw --frame-interval=30 --sel-policy=frame
```

### Example 2: Continuous Batching Mode

```powershell
.\flama.exe "D:\videos\sample.mp4" hw --use_cb --cb_batch_size=10 --new-multithread
```

### Example 3: Mixed Frame Selection Strategy

```powershell
.\flama.exe "D:\videos\sample.mp4" hw `
  --sel-policy=mixed `
  --frame-interval=30 `
  --window-seconds=2.0 `
  --max-per-window=3
```

### Example 4: Using JSON Config

```powershell
.\flama.exe "D:\videos\sample.mp4" hw --config=..\config\my_config.json
```

## Project Structure

```
flama/
├── CMakeLists.txt                 # CMake build configuration
├── vcpkg.json                     # vcpkg dependency manifest
├── vcpkg-configuration.json       # vcpkg configuration
├── README.md                      # This file
├── src/                           # Source code
│   ├── main.cpp                   # Main entry point (ffmpeg_interop)
│   ├── frame_selector.cpp/h       # Frame selection strategies
│   ├── vpp.cpp/h                  # Video post-processing (oneVPL)
│   ├── vlm_chat.cpp/h             # VLM scene understanding
│   ├── continuous_batching_chat.cpp/h  # Continuous batching
│   ├── video_segment.cpp/h        # Video segmentation
│   ├── hw_device_d3d11.cpp/h      # D3D11 device management
│   ├── texture_resource_pool.cpp/h # GPU texture pool
│   ├── parse_options.cpp/h        # Command-line parsing
│   ├── json_config.cpp/hpp        # JSON config loading
│   ├── util.cpp/h/hpp             # Utility functions
│   ├── profiling.h                # Performance profiling
│   ├── defs.h                     # Common definitions
│   └── ...                        # Other source files
├── config/                        # Configuration files
│   └── example_config.json        # Example configuration
├── build/                         # Build output (generated)
└── thirdparty/                    # Third-party dependencies (managed by vcpkg)
```

## Configuration

### Frame Selection Policies

- **frame**: Sample every N frames (use `--frame-interval`)
- **time**: Sample based on time windows (use `--window-seconds`, `--max-per-window`)
- **mixed**: Combine frame interval and time window strategies
- **key**: Prioritize keyframes with supplemental selection
- **mixed-key**: Force keyframes + mixed strategy

### Batch Processing Modes

- **VLM Pipeline**: Synchronous single-frame or multi-frame inference
- **Continuous Batching**: Asynchronous batch inference for higher throughput

## Performance Profiling

Flama outputs detailed performance statistics in CSV format:

- **Frame-level CSV**: Per-frame timing for each pipeline stage
- **Batch-level CSV**: Aggregated statistics per batch

Output files are saved in the output directory specified (default: current directory).

## Troubleshooting

### CMake Cannot Find Packages

If CMake cannot find OpenVINO, oneVPL, or other dependencies:

1. Verify environment variables are set correctly
2. Add package paths to CMAKE_PREFIX_PATH:

```powershell
cmake .. -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/openvino_2025;C:/Program Files/Intel/oneVPL" `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
```

### vcpkg Install Fails

If vcpkg fails to install dependencies:

1. Update vcpkg: `cd C:\vcpkg && git pull && .\bootstrap-vcpkg.bat`
2. Clear vcpkg cache: `vcpkg remove --outdated`
3. Reinstall: `vcpkg install --triplet=x64-windows`

### Runtime DLL Not Found

If the executable fails to run due to missing DLLs:

1. Ensure OpenVINO, oneVPL, and FFmpeg bin directories are in PATH
2. Copy required DLLs to the executable directory
3. Use `dumpbin /dependents flama.exe` to check dependencies

### Hardware Decode Not Working

If hardware decode fails:

1. Verify GPU supports D3D11VA
2. Update GPU drivers
3. Try software decode mode (`sw`) instead
4. Check FFmpeg was built with D3D11VA support

## License

This project includes code under various open source licenses. See individual source files for details.

## Acknowledgments

- Intel OpenVINO and oneVPL teams
- FFmpeg community
- Khronos OpenCL working group
- nlohmann for the JSON library

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style conventions
2. All dependencies are managed via vcpkg where possible
3. CMake configuration is updated for new dependencies
4. Documentation is updated accordingly

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review OpenVINO and oneVPL documentation
3. Open an issue on the project repository
