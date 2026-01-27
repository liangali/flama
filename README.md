# FLAMA - Fast Library for AI Multimodal Acceleration

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

The following dependencies are bundled in the repository:

- **Intel oneVPL**: Video post-processing (located in `thirdparty/_vplinstall/`)

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

### 2. Install OpenVINO and OpenVINO GenAI

Download the pre-built archives and extract them:

1. **OpenVINO Runtime**: Download from [OpenVINO Archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/) or [GitHub Releases](https://github.com/openvinotoolkit/openvino/releases)
2. **OpenVINO GenAI**: Download from [GenAI Releases](https://github.com/openvinotoolkit/openvino.genai/releases)

Extract to your preferred location. Example paths used in this guide:

```
D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64
D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64
```

### 3. oneVPL (Bundled)

Intel oneVPL is **bundled with the repository** - no installation required.

Location: `thirdparty/_vplinstall/`

CMake will automatically find VPL via the `-DVPL_DIR` flag (see Build section).

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

### Quick Build (Automated)

The project includes an automated build script `build.bat` that handles all configuration and compilation steps:

```cmd
# Navigate to project directory
cd D:\code\flama_code\flama

# Run the build script (handles CMake configuration and build)
.\build.bat
```

The script will:
1. Validate all dependency paths (OpenVINO, GenAI, TBB, VPL)
2. Configure CMake with Visual Studio 2022
3. Build the project in Release mode
4. Output the executable to: `build\bin\Release\flama.exe`

**Note**: Edit the path definitions at the top of `build.bat` if your dependencies are installed in different locations.

### Manual Build (Without Script)

If you prefer manual control, use CMake directly:

```powershell
cd D:\code\flama_code\flama
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake" `
  -DOpenVINO_DIR="D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\cmake" `
  -DOpenVINOGenAI_DIR="D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\cmake" `
  -DTBB_DIR="D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\lib\cmake\TBB" `
  -DVPL_DIR="%cd%\thirdparty\_vplinstall\lib\cmake\vpl"

cmake --build . --config Release
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
