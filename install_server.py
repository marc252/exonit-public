#!/usr/bin/env python3
"""ExoNit Server Installer - universal setup for Linux, macOS & Windows.

Usage:
    python install_server.py                          Interactive setup
    python install_server.py --update                 Update dependencies only
    python install_server.py --unattended             Non-interactive (defaults)
    python install_server.py --unattended --port 9000 --fits-dir /mnt/data/fits
"""

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import textwrap

# ── Constants ────────────────────────────────────────────────────────────────

MIN_PYTHON = (3, 9)
DEFAULT_PORT = 8400
DEFAULT_FITS_DIR = os.path.join("data", "fits_cache")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _enable_ansi_windows():
    """Enable ANSI escape codes on Windows 10+ via kernel32."""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
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
    print(bold("  ExoNit Server Installer"))
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
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 4,
    }


# ── Python validation ────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required "
             f"(found {v.major}.{v.minor}.{v.micro})")
        if platform.system() == "Windows":
            print(f"    Install with:  winget install Python.Python.3.12")
        elif platform.system() == "Darwin":
            print(f"    Install with:  brew install python")
        else:
            print(f"    Install with:  sudo apt install python3 python3-venv python3-pip")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")


# ── Virtual environment ──────────────────────────────────────────────────────

def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_venv_dir():
    return os.path.join(get_project_dir(), "venv")

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

def pip_install_requirements():
    """Install requirements.txt into the venv."""
    py = get_venv_python()
    req_file = os.path.join(get_project_dir(), "requirements.txt")
    if not os.path.isfile(req_file):
        fail("requirements.txt not found")
        sys.exit(1)
    step("Installing dependencies from requirements.txt...")
    cmd = [py, "-m", "pip", "install", "-q", "--upgrade", "-r", req_file]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        ok("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        fail(f"pip install failed (exit code {e.returncode})")
        return False

def pip_install_extra():
    """Install python-dotenv (needed by run_nightly.py)."""
    py = get_venv_python()
    step("Installing python-dotenv...")
    cmd = [py, "-m", "pip", "install", "-q", "--upgrade", "python-dotenv"]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        ok("python-dotenv installed")
    except subprocess.CalledProcessError:
        warn("python-dotenv install failed (optional, .env loading may not work)")


# ── Configuration ────────────────────────────────────────────────────────────

def load_existing_env():
    """Load existing .env file as a dict."""
    env_path = os.path.join(get_project_dir(), ".env")
    env = {}
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip().strip('"').strip("'")
    return env

def configure_interactive():
    """Interactive configuration wizard."""
    existing = load_existing_env()

    print()
    print(bold("  --- Configuration ---"))
    print()

    default_fits = existing.get("EXONIT_FITS_DIR", DEFAULT_FITS_DIR)
    fits_dir = ask("FITS data directory", default_fits)

    default_port = existing.get("EXONIT_PORT", str(DEFAULT_PORT))
    port = ask("Server port", default_port)

    default_token = existing.get("EXONIT_WORKER_TOKEN", "exonit-default-token")
    worker_token = ask("Worker auth token", default_token)

    # Anthropic API key (optional)
    default_api_key = existing.get("ANTHROPIC_API_KEY", "")
    api_key_prompt = "Anthropic API key (optional, for AI analysis)"
    if default_api_key:
        api_key = ask(api_key_prompt, default_api_key[:8] + "...")
        if api_key == default_api_key[:8] + "...":
            api_key = default_api_key
    else:
        api_key = ask(api_key_prompt, "")

    return {
        "EXONIT_PORT": port,
        "EXONIT_FITS_DIR": fits_dir,
        "EXONIT_WORKER_TOKEN": worker_token,
        "ANTHROPIC_API_KEY": api_key,
    }

def configure_unattended(args):
    """Non-interactive configuration from CLI args."""
    existing = load_existing_env()
    return {
        "EXONIT_PORT": args.port or existing.get("EXONIT_PORT", str(DEFAULT_PORT)),
        "EXONIT_FITS_DIR": args.fits_dir or existing.get("EXONIT_FITS_DIR", DEFAULT_FITS_DIR),
        "EXONIT_WORKER_TOKEN": args.worker_token or existing.get("EXONIT_WORKER_TOKEN", "exonit-default-token"),
        "ANTHROPIC_API_KEY": existing.get("ANTHROPIC_API_KEY", ""),
    }

def save_env(config):
    """Write .env file, preserving any extra keys from the existing file."""
    env_path = os.path.join(get_project_dir(), ".env")

    # Load existing keys we don't manage
    existing = load_existing_env()
    for key in existing:
        if key not in config:
            config[key] = existing[key]

    lines = ["# ExoNit Server Configuration (auto-generated by install_server.py)"]
    for key, val in config.items():
        if val:
            lines.append(f'{key}={val}')
    lines.append("")  # trailing newline

    with open(env_path, "w") as f:
        f.write("\n".join(lines))
    ok(f".env saved ({env_path})")

    # Create FITS directory if it doesn't exist
    fits_dir = config.get("EXONIT_FITS_DIR", DEFAULT_FITS_DIR)
    if not os.path.isabs(fits_dir):
        fits_dir = os.path.join(get_project_dir(), fits_dir)
    os.makedirs(fits_dir, exist_ok=True)
    ok(f"FITS directory ready ({fits_dir})")


# ── Service installation ────────────────────────────────────────────────────

def offer_service_install(plat, config):
    """Optionally install as a system service."""
    if plat["os"] == "Linux":
        print()
        answer = ask("Install as systemd service? (y/n)", "n")
        if answer.lower() == "y":
            install_systemd_service(config)
    elif plat["os"] == "Darwin":
        print()
        answer = ask("Install as launchd service (start on login)? (y/n)", "n")
        if answer.lower() == "y":
            install_launchd_service(config)
    elif plat["os"] == "Windows":
        print()
        answer = ask("Add to Windows startup? (y/n)", "n")
        if answer.lower() == "y":
            install_windows_startup()

def install_systemd_service(config):
    project_dir = get_project_dir()
    py = get_venv_python()
    port = config.get("EXONIT_PORT", str(DEFAULT_PORT))

    unit = textwrap.dedent(f"""\
        [Unit]
        Description=ExoNit Exoplanet Detection Server
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory={project_dir}
        ExecStart={py} run_nightly.py serve
        EnvironmentFile={project_dir}/.env
        Restart=on-failure
        RestartSec=10

        [Install]
        WantedBy=multi-user.target
    """)
    unit_path = os.path.expanduser("~/.config/systemd/user/exonit.service")
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "exonit"], check=False)
    ok(f"systemd user service installed ({unit_path})")
    print(f"    Start:   systemctl --user start exonit")
    print(f"    Status:  systemctl --user status exonit")
    print(f"    Logs:    journalctl --user -u exonit -f")
    print(f"    Stop:    systemctl --user stop exonit")

def install_launchd_service(config):
    project_dir = get_project_dir()
    py = get_venv_python()
    label = "com.exonit.server"

    # Build environment dict entries for the plist
    env_entries = []
    for key, val in config.items():
        if val:
            env_entries.append(f"            <key>{key}</key>")
            env_entries.append(f"            <string>{val}</string>")
    env_xml = "\n".join(env_entries)

    plist = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{label}</string>
            <key>WorkingDirectory</key>
            <string>{project_dir}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{py}</string>
                <string>run_nightly.py</string>
                <string>serve</string>
            </array>
            <key>EnvironmentVariables</key>
            <dict>
{env_xml}
            </dict>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{project_dir}/server.log</string>
            <key>StandardErrorPath</key>
            <string>{project_dir}/server.log</string>
        </dict>
        </plist>
    """)
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
    with open(plist_path, "w") as f:
        f.write(plist)
    ok(f"launchd service installed ({plist_path})")
    print(f"    Start:   launchctl load {plist_path}")
    print(f"    Stop:    launchctl unload {plist_path}")

def install_windows_startup():
    """Copy start_server.bat to Windows Startup folder."""
    project_dir = get_project_dir()
    bat_src = os.path.join(project_dir, "start_server.bat")
    startup_dir = os.path.join(
        os.environ.get("APPDATA", ""),
        "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    if not os.path.isdir(startup_dir):
        warn(f"Startup folder not found: {startup_dir}")
        return
    bat_dst = os.path.join(startup_dir, "exonit_server.bat")
    try:
        import shutil
        shutil.copy2(bat_src, bat_dst)
        ok(f"Added to Windows startup ({bat_dst})")
    except Exception as e:
        fail(f"Could not copy to startup: {e}")


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(plat, config):
    port = config.get("EXONIT_PORT", str(DEFAULT_PORT))
    fits_dir = config.get("EXONIT_FITS_DIR", DEFAULT_FITS_DIR)

    print()
    print(bold("============================================================"))
    print(bold("  Setup complete!"))
    print(bold("============================================================"))
    print()
    print(f"  Port:       {bold(port)}")
    print(f"  FITS dir:   {bold(fits_dir)}")
    print(f"  Dashboard:  {green(f'http://localhost:{port}')}")
    print(f"  API docs:   {green(f'http://localhost:{port}/docs')}")
    print()
    if plat["os"] == "Windows":
        print(f"  Start the server:  {green('start_server.bat')}")
    else:
        print(f"  Start the server:  {green('./start_server.sh')}")
    print(f"  Reconfigure:       python install_server.py")
    print(f"  Update deps:       python install_server.py --update")
    print()
    print(f"  {cyan('Next steps:')}")
    print(f"    1. Start the server")
    print(f"    2. Open http://localhost:{port} in your browser")
    print(f"    3. Set up workers: python worker/install.py")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ExoNit Server Installer")
    p.add_argument("--update", action="store_true",
                   help="Update dependencies only (skip config)")
    p.add_argument("--unattended", action="store_true",
                   help="Non-interactive mode (use defaults or flags)")
    p.add_argument("--port", default="",
                   help=f"Server port (default: {DEFAULT_PORT})")
    p.add_argument("--fits-dir", dest="fits_dir", default="",
                   help=f"FITS data directory (default: {DEFAULT_FITS_DIR})")
    p.add_argument("--worker-token", dest="worker_token", default="",
                   help="Worker authentication token")
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

    # 3. Dependencies
    if not pip_install_requirements():
        fail("Could not install dependencies")
        sys.exit(1)
    pip_install_extra()

    # Update-only mode stops here
    if args.update:
        print()
        ok("Dependencies updated")
        return

    # 4. Configuration
    print()
    if args.unattended:
        config = configure_unattended(args)
    else:
        config = configure_interactive()
    save_env(config)

    # 5. Optional system service
    if not args.unattended:
        offer_service_install(plat, config)

    # 6. Summary
    print_summary(plat, config)

if __name__ == "__main__":
    main()
