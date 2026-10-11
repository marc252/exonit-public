"""Terminal UI module for ExoNit Worker.

Rich terminal output with progress bars and star info panels.
Graceful fallback to plain text if `rich` is not installed.
"""

import os
import sys
import platform
import threading
import time as _time
from datetime import datetime


class WorkerUI:
    """Terminal UI for the ExoNit worker."""

    def __init__(self, worker_name, gpu_info, cpu_info, fpp_backend, bls_backend):
        self.worker_name = worker_name
        self.gpu_info = gpu_info
        self.cpu_info = cpu_info
        self.fpp_backend = fpp_backend
        self.bls_backend = bls_backend
        self._stop_event = threading.Event()
        self._on_stop = None  # callback to signal worker.running = False
        self._current_progress = None
        self._current_task = None
        self._fpp_scenarios_done = []

        # Counters
        self._stars_done = 0
        self._start_time = _time.time()
        self._avg_time = 0.0

        # Detect whether stdout is a real TTY. When the worker is started as a
        # background process (e.g. PowerShell Start-Process -WindowStyle Hidden
        # on Windows, which can leave stdout unattached), calling print() on an
        # invalid stdout raises OSError [Errno 22] Invalid argument mid-task.
        # In that case, run in quiet mode: all UI output becomes a no-op and
        # progress updates are sent only to the logger (via worker.py, which
        # owns logging setup). The worker keeps processing normally.
        try:
            self._is_tty = bool(sys.stdout and sys.stdout.isatty())
        except (AttributeError, ValueError, OSError):
            self._is_tty = False
        self._quiet = not self._is_tty

        if self._quiet:
            self._use_rich = False
            self._console = None
        else:
            try:
                from rich.console import Console
                self._use_rich = True
                self._console = Console()
            except (ImportError, TypeError):
                self._use_rich = False
                self._console = None

    def _safe_print(self, *args, **kwargs):
        """print() that never raises — silently drops if stdout is dead."""
        if self._quiet:
            return
        try:
            print(*args, **kwargs)
        except (OSError, ValueError):
            # stdout closed or redirected to a dead FD → switch to quiet mode
            self._quiet = True

    @property
    def stop_requested(self):
        return self._stop_event.is_set()

    # ── Banner ──────────────────────────────────────────────

    def show_banner(self):
        """Display startup info panel."""
        if self._quiet:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        ram_str = ""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            ram_str = f"RAM: {ram_gb:.1f} GB  .  "
        except Exception:
            pass  # cosmetic banner only; psutil optional

        py_ver = platform.python_version()

        if self._use_rich:
            from rich.panel import Panel
            content = (
                f"  {self.worker_name}  .  started {now}\n"
                f"  CPU: {self.cpu_info}\n"
                f"  GPU: {self.gpu_info}\n"
                f"  FPP: {self.fpp_backend}  .  BLS: {self.bls_backend}\n"
                f"  {ram_str}Python {py_ver}\n"
                f"  Press [bold][Q][/bold] to stop gracefully"
            )
            panel = Panel(content, title="ExoNit Worker", border_style="blue")
            self._console.print(panel)
        else:
            print(f"+-- ExoNit Worker --------+")
            print(f"|  {self.worker_name}  .  started {now}")
            print(f"|  CPU: {self.cpu_info}")
            print(f"|  GPU: {self.gpu_info}")
            print(f"|  FPP: {self.fpp_backend}  .  BLS: {self.bls_backend}")
            print(f"|  {ram_str}Python {py_ver}")
            print(f"|  Press [Q] to stop gracefully")
            print(f"+-------------------------+")

    # ── Star output ─────────────────────────────────────────

    def _stats_line(self):
        """Build inline stats: stars done, uptime, avg."""
        uptime = _time.time() - self._start_time
        if uptime < 3600:
            uptime_str = f"{uptime / 60:.0f}m"
        else:
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
        parts = [f"Stars processed: {self._stars_done}"]
        if self._stars_done > 0:
            parts.append(f"Avg: {self._avg_time:.1f}s/star")
        parts.append(f"Uptime: {uptime_str}")
        return "  .  ".join(parts)

    def star_start(self, tic_id, ra, dec, tmag, sectors, spectral, mass, radius):
        if self._quiet:
            return
        if isinstance(sectors, list):
            sectors_str = ", ".join(str(s) for s in sectors)
        else:
            sectors_str = str(sectors)
        exofop = f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"
        stats = self._stats_line()

        if self._use_rich:
            from rich.panel import Panel
            lines = [f"  RA {ra}  .  Dec {dec}  .  Tmag {tmag}"]
            if sectors_str:
                lines.append(f"  Sectors {sectors_str}  .  {radius} Rsun")
            lines.append(f"  [link={exofop}]{exofop}[/link]")
            lines.append(f"  [dim]{stats}[/dim]")
            panel = Panel("\n".join(lines), title=f"* TIC {tic_id}", border_style="cyan")
            self._console.print(panel)
        else:
            print(f"\n--- TIC {tic_id} ---")
            print(f"  RA {ra}  .  Dec {dec}  .  Tmag {tmag}")
            if sectors_str:
                print(f"  Sectors {sectors_str}  .  {radius} Rsun")
            print(f"  {exofop}")
            print(f"  {stats}")

    def phase_start(self, name):
        self._fpp_scenarios_done = []
        if self._quiet:
            return
        if self._use_rich:
            from rich.progress import (
                Progress, TextColumn, BarColumn,
                TaskProgressColumn, TimeElapsedColumn,
            )
            self._current_progress = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self._console,
            )
            self._current_task = self._current_progress.add_task(
                name.ljust(20), total=1000)
            self._current_progress.start()
        else:
            print(f"  >> {name}")

    def phase_update(self, fraction, status_text=None):
        if self._quiet:
            return
        if self._use_rich and self._current_progress is not None:
            completed = int(fraction * 1000)
            self._current_progress.update(self._current_task, completed=completed)
            if status_text is not None:
                self._current_progress.update(
                    self._current_task, description=status_text.ljust(20))
        elif status_text is not None:
            print(f"     {status_text}")

    def fpp_scenario_done(self, scenario_name):
        self._fpp_scenarios_done.append(scenario_name)
        desc = f"FPP [{', '.join(self._fpp_scenarios_done)}]".ljust(20)
        if self._use_rich and self._current_progress is not None:
            self._current_progress.update(self._current_task, description=desc)

    def phase_done(self, elapsed_seconds, suffix=""):
        if self._quiet:
            return
        if self._use_rich and self._current_progress is not None:
            self._current_progress.update(self._current_task, completed=1000)
            self._current_progress.stop()
            self._current_progress = None
            self._current_task = None
        else:
            msg = f"     done in {elapsed_seconds:.1f}s"
            if suffix:
                msg += f" {suffix}"
            print(msg)

    def star_result(self, candidates, period, depth, fpp, nfpp, verdict, elapsed):
        if self._quiet:
            return
        if self._use_rich:
            from rich.panel import Panel
            verdict_labels = {
                "validated": ("green", "Validated planet"),
                "candidate": ("yellow", "Planet candidate"),
                "likely_fp": ("red", "False positive"),
                "no_transit": ("dim", "No transit detected"),
            }
            color, label = verdict_labels.get(verdict, ("white", verdict))
            content = (
                f"  Candidates: {candidates}  .  Period: {period}  .  Depth: {depth}\n"
                f"  FPP: {fpp}  .  NFPP: {nfpp}\n"
                f"  [{color} bold]{label}[/{color} bold]  .  {elapsed:.1f}s"
            )
            panel = Panel(content, title="Result", border_style=color)
            self._console.print(panel)
        else:
            print(f"  Result: {verdict} | candidates={candidates} period={period} "
                  f"depth={depth} FPP={fpp} NFPP={nfpp} ({elapsed:.1f}s)")

    def update_footer(self, stars_done, uptime_seconds, avg_time):
        self._stars_done = stars_done
        self._avg_time = avg_time

    def waiting(self):
        if self._quiet:
            return
        if self._use_rich:
            self._console.print("  [dim]Waiting for tasks...[/dim]")
        else:
            print("  Waiting for tasks...")

    def request_stop(self):
        if self._stop_event.is_set():
            # Second Q press: force exit
            print("\n  Force exit!")
            os._exit(1)
        self._stop_event.set()
        if getattr(self, '_on_stop', None):
            self._on_stop()
        msg = "Stop requested -- will finish current star and exit. Press Q again to force quit."
        if self._use_rich and self._console is not None:
            self._console.print(f"  [bold yellow]{msg}[/bold yellow]")
        else:
            print(f"  {msg}")
        return True

    def cleanup(self):
        """Nothing to reset without scroll regions."""
        pass

    # ── Background threads ──────────────────────────────────

    def start_keypress_listener(self):
        threading.Thread(target=self._keypress_loop, daemon=True).start()

    def _keypress_loop(self):
        if sys.platform == "win32":
            self._keypress_loop_windows()
        else:
            self._keypress_loop_unix()

    def _keypress_loop_windows(self):
        try:
            import msvcrt
        except ImportError:
            return
        while not self._stop_event.is_set():
            try:
                if msvcrt.kbhit():
                    if msvcrt.getch() in (b'q', b'Q'):
                        self.request_stop()
                        return
            except Exception:
                return
            _time.sleep(0.1)

    def _keypress_loop_unix(self):
        try:
            import termios, select, tty
        except ImportError:
            return
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except Exception:
            return
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    if sys.stdin.read(1) in ('q', 'Q'):
                        self.request_stop()
                        return
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass
