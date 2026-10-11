# tests/test_worker_ui.py
"""Tests for WorkerUI terminal output module."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker"))


class TestWorkerUIPlainFallback:
    """Test that UI works without rich installed."""

    def test_create_ui_without_rich(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'rich', None)
        monkeypatch.setitem(sys.modules, 'rich.console', None)
        monkeypatch.setitem(sys.modules, 'rich.panel', None)
        monkeypatch.setitem(sys.modules, 'rich.progress', None)
        monkeypatch.setitem(sys.modules, 'rich.live', None)
        monkeypatch.setitem(sys.modules, 'rich.table', None)
        if 'worker_ui' in sys.modules:
            del sys.modules['worker_ui']
        from worker_ui import WorkerUI
        ui = WorkerUI("test-worker", "RTX 4070", "Ryzen 7",
                       "cuda", "cuda")
        assert ui._use_rich is False

    @staticmethod
    def _bare_ui():
        """Build a WorkerUI that skips rich probing in __init__ but has all
        the attributes __init__ would have set. Used to test plain-text
        code paths without also importing rich."""
        import threading
        import time as _time
        from worker_ui import WorkerUI
        ui = WorkerUI.__new__(WorkerUI)
        ui._use_rich = False
        ui._quiet = False
        ui._console = None
        ui._stop_event = threading.Event()
        ui._on_stop = None
        ui._current_progress = None
        ui._current_task = None
        ui._fpp_scenarios_done = []
        ui._stars_done = 0
        ui._start_time = _time.time()
        ui._avg_time = 0.0
        return ui

    def test_show_banner_plain(self, capsys):
        ui = self._bare_ui()
        ui.worker_name = "test-worker"
        ui.gpu_info = "RTX 4070"
        ui.cpu_info = "8 cores"
        ui.fpp_backend = "cuda"
        ui.bls_backend = "cuda"
        ui.show_banner()
        captured = capsys.readouterr()
        assert "test-worker" in captured.out

    def test_star_start_plain(self, capsys):
        ui = self._bare_ui()
        ui.star_start("261136679", "12h34m56s", "+45d12m34s",
                       11.2, [14, 15], "G5V", 0.92, 0.88)
        captured = capsys.readouterr()
        assert "261136679" in captured.out

    def test_request_stop_returns_true(self):
        ui = self._bare_ui()
        result = ui.request_stop()
        assert result is True
        assert ui._stop_event.is_set()


class TestWorkerUIRich:
    """Test rich UI if available."""

    def test_create_ui_with_rich(self, monkeypatch):
        try:
            import rich
        except ImportError:
            import pytest
            pytest.skip("rich not installed")
        # Under pytest, stdout is captured (not a TTY) so WorkerUI would go
        # quiet and skip rich. Force isatty() → True to exercise the rich path.
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
        from worker_ui import WorkerUI
        ui = WorkerUI("test-worker", "RTX 4070", "Ryzen 7 5800X",
                       "cuda", "cuda")
        assert ui._use_rich is True

    def test_phase_start_and_done(self, monkeypatch):
        try:
            import rich
        except ImportError:
            import pytest
            pytest.skip("rich not installed")
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
        from worker_ui import WorkerUI
        ui = WorkerUI("test-worker", "RTX 4070", "Ryzen 7",
                       "cuda", "cuda")
        ui.phase_start("BLS search")
        ui.phase_update(0.5)
        ui.phase_done(3.2)
