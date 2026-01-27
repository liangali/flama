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

#### Directory Structure

Both packages have the same structure:

```
<package_root>/
├── setupvars.bat          # Environment initialization script
├── setupvars.ps1          # PowerShell variant
├── runtime/
│   ├── cmake/             # CMake config files (OpenVINOConfig.cmake, OpenVINOGenAIConfig.cmake)
│   ├── bin/intel64/Release/  # DLLs (openvino.dll, openvino_genai.dll, plugins, etc.)
│   ├── include/openvino/    # Header files
│   └── 3rdparty/tbb/      # TBB dependency (OpenVINO only)
├── python/                # Python bindings
└── samples/               # Example code
```

#### Option A: Use setupvars.bat (Per-Session)

Run before each build/run session:

```cmd
:: Initialize OpenVINO environment
call "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\setupvars.bat"

:: Initialize GenAI environment (run AFTER OpenVINO setupvars)
call "D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\setupvars.bat"
```

This sets:
- `OpenVINO_DIR` → `<openvino_root>\runtime\cmake`
- `OpenVINOGenAI_DIR` → `<genai_root>\runtime\cmake`
- `TBB_DIR` → `<openvino_root>\runtime\3rdparty\tbb\lib\cmake\TBB`
- `PATH` includes DLL directories

#### Option B: Set Persistent Environment Variables

Set these environment variables permanently (adjust paths to your installation):

```powershell
# OpenVINO CMake config directory
setx OpenVINO_DIR "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\cmake"

# OpenVINO GenAI CMake config directory
setx OpenVINOGenAI_DIR "D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\cmake"

# TBB CMake config directory
setx TBB_DIR "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\lib\cmake\TBB"
```

Add DLL directories to PATH:

```powershell
# Add OpenVINO and GenAI DLLs to PATH
setx PATH "%PATH%;D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\bin\intel64\Release;D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\bin\intel64\Release;D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\bin"
```

**Note**: Close and reopen your terminal after setting environment variables with `setx`.

#### Required DLLs at Runtime

From **OpenVINO** (`runtime/bin/intel64/Release/`):
- `openvino.dll` - Core runtime
- `openvino_intel_gpu_plugin.dll` - GPU inference (for `device="GPU"`)
- `openvino_intel_cpu_plugin.dll` - CPU inference fallback
- `openvino_auto_plugin.dll` - AUTO device selection
- Various frontend DLLs (onnx, pytorch, etc.)

From **OpenVINO GenAI** (`runtime/bin/intel64/Release/`):
- `openvino_genai.dll` - GenAI runtime
- `openvino_tokenizers.dll` - Tokenizer support
- `icudt70.dll`, `icuuc70.dll` - ICU unicode libraries

From **TBB** (`runtime/3rdparty/tbb/bin/`):
- `tbb12.dll` - Threading Building Blocks

### 3. oneVPL (Bundled)

Intel oneVPL is **bundled with the repository** - no installation required.

Location: `thirdparty/_vplinstall/`

```
thirdparty/_vplinstall/
├── bin/                    # libvpl.dll and runtime DLLs
├── include/vpl/            # Headers (mfx*.h)
├── lib/
│   ├── vpl.lib             # Import library
│   └── cmake/vpl/          # VPLConfig.cmake
└── share/vpl/              # Examples and licensing
```

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

### Prerequisites Check

Before building, ensure you have initialized the OpenVINO environment. Choose one method:

**Method 1: Run setupvars (recommended for first-time setup)**
```cmd
call "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\setupvars.bat"
call "D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\setupvars.bat"
```

**Method 2: Verify environment variables are set**
```powershell
echo $env:OpenVINO_DIR      # Should show: ...\runtime\cmake
echo $env:OpenVINOGenAI_DIR # Should show: ...\runtime\cmake
echo $env:TBB_DIR           # Should show: ...\3rdparty\tbb\lib\cmake\TBB
```

### Option 1: Using Environment Variables (Recommended)

If you have set `OpenVINO_DIR`, `OpenVINOGenAI_DIR`, and `TBB_DIR` environment variables (via setupvars.bat or setx):

```powershell
# Navigate to project directory
cd D:\code\flama_code\flama

# Create and enter build directory
mkdir build
cd build

# Configure CMake - it will find OpenVINO automatically via environment variables
# VPL is bundled in thirdparty, specify its path explicitly
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVPL_DIR="$PWD\..\thirdparty\_vplinstall\lib\cmake\vpl"

# Build the project
cmake --build . --config Release

# The executable will be in: build\bin\Release\flama.exe
```

### Option 2: Explicit CMake Paths

If environment variables are not set, specify paths directly:

```powershell
cd D:\code\flama_code\flama
mkdir build
cd build

# Configure with explicit paths for all dependencies
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DOpenVINO_DIR="D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\cmake" `
  -DOpenVINOGenAI_DIR="D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\cmake" `
  -DTBB_DIR="D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\lib\cmake\TBB" `
  -DVPL_DIR="D:\code\flama_code\flama\thirdparty\_vplinstall\lib\cmake\vpl"

cmake --build . --config Release
```

### Build Options

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

### CMake Cannot Find OpenVINO or OpenVINO GenAI

If CMake cannot find OpenVINO packages:

1. **Run setupvars.bat** before configuring CMake:
   ```cmd
   call "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\setupvars.bat"
   call "D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\setupvars.bat"
   ```

2. **Or specify paths explicitly** in cmake command:
   ```powershell
   cmake .. -G "Visual Studio 17 2022" -A x64 `
     -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
     -DOpenVINO_DIR="D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\cmake" `
     -DOpenVINOGenAI_DIR="D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\cmake" `
     -DTBB_DIR="D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\lib\cmake\TBB" `
     -DVPL_DIR="D:\code\flama_code\flama\thirdparty\_vplinstall\lib\cmake\vpl"
   ```

3. **Verify the cmake directory exists** and contains `OpenVINOConfig.cmake`:
   ```powershell
   dir "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\cmake"
   ```

### vcpkg Install Fails

If vcpkg fails to install dependencies:

1. Update vcpkg: `cd C:\vcpkg && git pull && .\bootstrap-vcpkg.bat`
2. Clear vcpkg cache: `vcpkg remove --outdated`
3. Reinstall: `vcpkg install --triplet=x64-windows`

### Runtime DLL Not Found

If the executable fails to run due to missing DLLs:

1. **Run setupvars.bat** before running the application:
   ```cmd
   call "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\setupvars.bat"
   call "D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\setupvars.bat"
   ```

2. **Or add DLL directories to PATH** permanently:
   ```powershell
   # OpenVINO DLLs
   D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\bin\intel64\Release

   # GenAI DLLs
   D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\bin\intel64\Release

   # TBB DLLs
   D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\bin

   # VPL DLLs (bundled)
   D:\code\flama_code\flama\thirdparty\_vplinstall\bin
   ```

3. **Or copy required DLLs** to the executable directory (`build\bin\Release\`)

4. **Check dependencies** using:
   ```cmd
   dumpbin /dependents build\bin\Release\flama.exe
   ```

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
