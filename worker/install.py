#!/usr/bin/env python3
"""ExoNit Worker Installer - universal setup for Windows, macOS, Linux & Raspberry Pi.

Usage:
    python install.py              Interactive setup
    python install.py --update     Update dependencies only
    python install.py --unattended --server-url URL --token TOKEN   Non-interactive
"""

import argparse
import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import textwrap

# ── Constants ────────────────────────────────────────────────────────────────

MIN_PYTHON = (3, 8)
DEFAULT_SERVER = "http://YOUR_SERVER:8400"
DEFAULT_TOKEN = "exonit-default-token"

BASE_DEPS = ["numpy", "scipy", "requests", "onnxruntime", "lightkurve", "rich", "numba"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _enable_ansi_windows():
    """Enable ANSI escape codes on Windows 10+ via kernel32."""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11, ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

_enable_ansi_windows()

def _c(text, code):
    """Colorize text if terminal supports it."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _c(t, 32)
def yellow(t): return _c(t, 33)
def red(t):    return _c(t, 31)
def bold(t):   return _c(t, 1)
def cyan(t):   return _c(t, 36)

def banner():
    print()
    print(bold("============================================================"))
    print(bold("  ExoNit Worker Installer"))
    print(bold("============================================================"))
    print()

def step(msg):
    print(f"  {cyan('>>>')} {msg}")

def ok(msg):
    print(f"  {green('[OK]')} {msg}")

def warn(msg):
    print(f"  {yellow('[!]')} {msg}")

def fail(msg):
    print(f"  {red('[ERROR]')} {msg}")

def ask(prompt, default=""):
    """Prompt user for input with a default value."""
    if default:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"  {prompt}: ").strip()


# ── Platform detection ───────────────────────────────────────────────────────

def detect_platform():
    """Return a dict with platform info."""
    info = {
        "os": platform.system(),          # Windows, Darwin, Linux
        "arch": platform.machine(),        # x86_64, arm64, aarch64, armv7l
        "is_rpi": False,
        "cpu_count": os.cpu_count() or 4,
    }
    # Raspberry Pi detection
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().lower()
        if "raspberry" in model:
            info["is_rpi"] = True
    except (FileNotFoundError, PermissionError):
        pass
    return info


# ── Python validation ────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required (found {v.major}.{v.minor}.{v.micro})")
        if platform.system() == "Windows":
            print(f"    Install with:  winget install Python.Python.3.12")
        elif platform.system() == "Darwin":
            print(f"    Install with:  brew install python")
        else:
            print(f"    Install with:  sudo apt install python3 python3-venv python3-pip")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")


# ── Virtual environment ──────────────────────────────────────────────────────

def get_venv_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")

def get_venv_python():
    venv = get_venv_dir()
    if platform.system() == "Windows":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python3")

def create_venv():
    venv = get_venv_dir()
    if os.path.isdir(venv):
        ok("Virtual environment exists")
        return
    step("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", venv],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok("Virtual environment created")

def pip_install(packages, label=None):
    """Install packages into the venv."""
    py = get_venv_python()
    cmd = [py, "-m", "pip", "install", "-q", "--upgrade"] + packages
    display = label or ", ".join(packages)
    step(f"Installing {display}...")
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


# ── GPU detection ────────────────────────────────────────────────────────────

def detect_gpu(plat):
    """Detect available GPU and install corresponding packages.

    Returns (gpu_enabled: bool, gpu_info: str).
    """
    # 1. NVIDIA CUDA
    if shutil.which("nvidia-smi"):
        step("NVIDIA GPU detected - installing CUDA acceleration...")
        if pip_install(["cupy-cuda12x"], "cupy-cuda12x"):
            ok("CUDA acceleration ready")
            return True, "NVIDIA CUDA"
        else:
            warn("cupy-cuda12x install failed  - falling back")

    # 2. Apple Silicon (Metal / MLX)
    if plat["os"] == "Darwin" and plat["arch"] == "arm64":
        step("Apple Silicon detected  - installing MLX...")
        if pip_install(["mlx"], "mlx"):
            ok("Metal/MLX acceleration ready")
            return True, "Apple Silicon (Metal/MLX)"
        else:
            warn("MLX install failed  - falling back")

    # 3. OpenCL (AMD / Intel iGPU on Linux)
    if plat["os"] == "Linux" and not plat["is_rpi"]:
        has_opencl = any(
            os.path.exists(p) for p in [
                "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
                "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
                "/usr/lib/libOpenCL.so",
                "/usr/lib/libOpenCL.so.1",
            ]
        )
        if has_opencl:
            step("OpenCL available  - installing pyopencl...")
            if pip_install(["pyopencl"], "pyopencl"):
                ok("OpenCL acceleration ready")
                return True, "OpenCL (AMD/Intel)"
            else:
                warn("pyopencl install failed  - falling back")

    # 4. Raspberry Pi
    if plat["is_rpi"]:
        ok("Raspberry Pi  - CPU mode")
        return False, "Raspberry Pi (CPU)"

    # 5. CPU fallback
    ok("No GPU detected  - CPU mode")
    return False, "CPU only"


# ── Configuration ────────────────────────────────────────────────────────────

def load_existing_config():
    """Load existing config.json if present."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def test_server_connection(url):
    """Test if the server is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/api/version", method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        return True, data.get("version", "unknown")
    except Exception:
        return False, None

def configure_interactive(plat, gpu_enabled, gpu_info):
    """Interactive configuration wizard."""
    existing = load_existing_config()

    print()
    print(bold("  --- Configuration ---"))
    print()

    server_url = ask("Server URL", existing.get("server_url", DEFAULT_SERVER))
    token = ask("Worker token", existing.get("token", DEFAULT_TOKEN))

    default_name = existing.get("worker_name", f"worker-{socket.gethostname()}")
    worker_name = ask("Worker name", default_name)

    # Test connection
    step(f"Testing connection to {server_url}...")
    reachable, server_version = test_server_connection(server_url)
    if reachable:
        ok(f"Server reachable (v{server_version})")
    else:
        warn("Server not reachable  - config saved anyway, connect later")

    config = {
        "server_url": server_url,
        "token": token,
        "worker_name": worker_name,
        "cpu_threads": plat["cpu_count"],
        "gpu_enabled": gpu_enabled,
        "poll_interval_seconds": 5,
        "heartbeat_interval_seconds": 30,
        "fits_cache_dir": "fits_cache",
        "gui_port": 8401,
        "log_file": "worker.log",
        "log_level": "INFO",
    }

    # Preserve extra keys from existing config (e.g. fpp_cpu_cores)
    for key in existing:
        if key not in config:
            config[key] = existing[key]

    return config

def configure_unattended(args, plat, gpu_enabled):
    """Non-interactive configuration from CLI args."""
    existing = load_existing_config()
    return {
        "server_url": args.server_url or existing.get("server_url", DEFAULT_SERVER),
        "token": args.token or existing.get("token", DEFAULT_TOKEN),
        "worker_name": args.worker_name or existing.get("worker_name", f"worker-{socket.gethostname()}"),
        "cpu_threads": plat["cpu_count"],
        "gpu_enabled": gpu_enabled,
        "poll_interval_seconds": 5,
        "heartbeat_interval_seconds": 30,
        "fits_cache_dir": "fits_cache",
        "gui_port": 8401,
        "log_file": "worker.log",
        "log_level": "INFO",
    }

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    ok("Config saved to config.json")


# ── Launcher generation ──────────────────────────────────────────────────────

def generate_launchers():
    """Generate platform-appropriate start scripts."""
    worker_dir = os.path.dirname(os.path.abspath(__file__))

    # Unix launcher (Linux / macOS)
    sh_path = os.path.join(worker_dir, "start.sh")
    with open(sh_path, "w", newline="\n") as f:
        f.write(textwrap.dedent("""\
            #!/bin/bash
            cd "$(dirname "$0")"
            source venv/bin/activate
            python3 worker.py --config config.json
        """))
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass  # Windows can't chmod

    # Windows launcher
    bat_path = os.path.join(worker_dir, "start.bat")
    with open(bat_path, "w", newline="\r\n") as f:
        f.write(textwrap.dedent("""\
            @echo off
            title ExoNit Worker
            cd /d "%~dp0"
            call venv\\Scripts\\activate
            python worker.py --config config.json
            pause
        """))

    ok("Launchers generated (start.sh + start.bat)")


# ── Service installation (optional) ─────────────────────────────────────────

def offer_service_install(plat):
    """Optionally install as a system service."""
    if plat["os"] == "Linux" and not plat["is_rpi"]:
        print()
        answer = ask("Install as systemd service? (y/n)", "n")
        if answer.lower() == "y":
            install_systemd_service()
    elif plat["os"] == "Darwin":
        print()
        answer = ask("Install as launchd service (start on login)? (y/n)", "n")
        if answer.lower() == "y":
            install_launchd_service()

def install_systemd_service():
    worker_dir = os.path.dirname(os.path.abspath(__file__))
    py = get_venv_python()
    unit = textwrap.dedent(f"""\
        [Unit]
        Description=ExoNit Worker
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory={worker_dir}
        ExecStart={py} worker.py --config config.json
        Restart=on-failure
        RestartSec=10

        [Install]
        WantedBy=multi-user.target
    """)
    unit_path = os.path.expanduser("~/.config/systemd/user/exonit-worker.service")
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "exonit-worker"], check=False)
    ok("systemd user service installed")
    print(f"    Start:   systemctl --user start exonit-worker")
    print(f"    Status:  systemctl --user status exonit-worker")
    print(f"    Logs:    journalctl --user -u exonit-worker -f")

def install_launchd_service():
    worker_dir = os.path.dirname(os.path.abspath(__file__))
    py = get_venv_python()
    label = "com.exonit.worker"
    plist = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{label}</string>
            <key>WorkingDirectory</key>
            <string>{worker_dir}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{py}</string>
                <string>worker.py</string>
                <string>--config</string>
                <string>config.json</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{worker_dir}/worker.log</string>
            <key>StandardErrorPath</key>
            <string>{worker_dir}/worker.log</string>
        </dict>
        </plist>
    """)
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
    with open(plist_path, "w") as f:
        f.write(plist)
    ok("launchd service installed")
    print(f"    Start:   launchctl load {plist_path}")
    print(f"    Stop:    launchctl unload {plist_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ExoNit Worker Installer")
    p.add_argument("--update", action="store_true",
                   help="Update dependencies only (skip config)")
    p.add_argument("--unattended", action="store_true",
                   help="Non-interactive mode")
    p.add_argument("--server-url", dest="server_url", default="",
                   help="Server URL (unattended mode)")
    p.add_argument("--token", default="",
                   help="Worker token (unattended mode)")
    p.add_argument("--worker-name", dest="worker_name", default="",
                   help="Worker name (unattended mode)")
    return p.parse_args()

def main():
    args = parse_args()
    banner()

    # 1. Platform & Python
    step("Detecting platform...")
    plat = detect_platform()
    ok(f"{plat['os']} {plat['arch']} ({plat['cpu_count']} cores)")
    check_python()

    # 2. Virtual environment
    create_venv()

    # 3. Base dependencies
    if not pip_install(BASE_DEPS, "base dependencies"):
        fail("Could not install base dependencies")
        sys.exit(1)
    ok("Base dependencies installed")

    # 4. GPU
    print()
    step("Detecting GPU acceleration...")
    gpu_enabled, gpu_info = detect_gpu(plat)
    print()
    print(f"  GPU: {bold(gpu_info)}")

    # Update-only mode stops here
    if args.update:
        print()
        ok("Dependencies updated")
        return

    # 5. Configuration
    if args.unattended:
        config = configure_unattended(args, plat, gpu_enabled)
    else:
        config = configure_interactive(plat, gpu_enabled, gpu_info)
    save_config(config)

    # 6. Generate launchers
    generate_launchers()

    # 7. Optional service
    if not args.unattended:
        offer_service_install(plat)

    # 8. Done
    print()
    print(bold("============================================================"))
    print(bold("  Setup complete!"))
    print(bold("============================================================"))
    print()
    if plat["os"] == "Windows":
        print(f"  Start the worker:  {green('start.bat')}")
    else:
        print(f"  Start the worker:  {green('./start.sh')}")
    print(f"  Reconfigure:       python install.py")
    print(f"  Update deps:       python install.py --update")
    print()

if __name__ == "__main__":
    main()
