#!/bin/bash
# ExoNit Worker Setup — Linux / macOS / Raspberry Pi
set -e

echo "============================================================"
echo " ExoNit Worker Installer"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Install it first:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  macOS: brew install python"
    echo "  Raspberry Pi: sudo apt install python3 python3-venv"
    exit 1
fi
echo "[OK] Python3 found: $(python3 --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install base dependencies
echo "Installing dependencies..."
pip install -q numpy requests onnxruntime lightkurve

# Detect GPU
GPU_ENABLED=false
GPU_INFO="CPU only"

# Check NVIDIA CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] NVIDIA GPU detected"
    pip install -q cupy-cuda12x 2>/dev/null && GPU_ENABLED=true && GPU_INFO="NVIDIA CUDA" || true
fi

# Check Apple Silicon (Metal/MLX)
if [ "$GPU_ENABLED" = "false" ] && [ "$(uname)" = "Darwin" ]; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        echo "[OK] Apple Silicon detected — installing MLX..."
        pip install -q mlx 2>/dev/null && GPU_ENABLED=true && GPU_INFO="Apple Silicon (Metal/MLX)" || true
    else
        echo "[INFO] Intel Mac detected — no GPU acceleration"
    fi
fi

# Check AMD OpenCL (Linux)
if [ "$GPU_ENABLED" = "false" ] && [ -f /usr/lib/x86_64-linux-gnu/libOpenCL.so* ] 2>/dev/null; then
    echo "[INFO] OpenCL available (AMD/Intel iGPU)"
    pip install -q pyopencl 2>/dev/null && GPU_INFO="OpenCL" || true
fi

# Check Raspberry Pi (VideoCore)
if [ -f /proc/device-tree/model ] && grep -qi "raspberry" /proc/device-tree/model; then
    echo "[INFO] Raspberry Pi detected"
    GPU_INFO="Raspberry Pi CPU"
fi

echo
echo "GPU: $GPU_INFO (gpu_enabled=$GPU_ENABLED)"
echo

# Read existing config.json defaults if present
EXISTING_URL=""
EXISTING_TOKEN=""
EXISTING_NAME=""
if [ -f "config.json" ]; then
    EXISTING_URL=$(python3 -c "import json; print(json.load(open('config.json')).get('server_url',''))" 2>/dev/null)
    EXISTING_TOKEN=$(python3 -c "import json; print(json.load(open('config.json')).get('token',''))" 2>/dev/null)
    EXISTING_NAME=$(python3 -c "import json; print(json.load(open('config.json')).get('worker_name',''))" 2>/dev/null)
fi

# Configure (press Enter to keep existing values)
DEFAULT_URL=${EXISTING_URL:-"http://YOUR_SERVER:8400"}
DEFAULT_TOKEN=${EXISTING_TOKEN:-"exonit-default-token"}
DEFAULT_NAME=${EXISTING_NAME:-$(hostname)}

read -p "Server URL [$DEFAULT_URL]: " SERVER_URL
SERVER_URL=${SERVER_URL:-$DEFAULT_URL}
read -p "Worker token [$DEFAULT_TOKEN]: " TOKEN
TOKEN=${TOKEN:-$DEFAULT_TOKEN}
read -p "Worker name [$DEFAULT_NAME]: " WORKER_NAME
WORKER_NAME=${WORKER_NAME:-$DEFAULT_NAME}

cat > config.json <<EOF
{
    "server_url": "$SERVER_URL",
    "token": "$TOKEN",
    "worker_name": "$WORKER_NAME",
    "cpu_threads": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4),
    "gpu_enabled": $GPU_ENABLED,
    "poll_interval_seconds": 5,
    "heartbeat_interval_seconds": 30,
    "fits_cache_dir": "fits_cache",
    "gui_port": 8401
}
EOF

echo
echo "[OK] Setup complete!"
echo "  Config: config.json"
echo "  Start:  ./start_worker.sh"
echo
echo "Optional config flag (worker/config.json):"
echo "  ffi_fallback_on_load_fail: false  # set to true to recover load_failed TICs"
echo "                                     # via the server's Phase 2.5 FFI cache"
echo
