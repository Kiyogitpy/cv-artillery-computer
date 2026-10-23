"""pytest setup: keep the tests quiet.

winkeys.beep() plays a real tone through winsound, and artillery's F5/F8 handlers
call it, so the artillery tests used to beep out loud. Muted here for every test.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_beeps(monkeypatch):
    import artillery
    import winkeys
    monkeypatch.setattr(winkeys, "beep", lambda *a, **k: None)
    monkeypatch.setattr(artillery, "beep", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _collect_closed_windows():
    """Collect each test's closed Tk window on the main thread. Left to chance, the collector runs
    on whatever thread is allocating -- a minimap worker, an F5 or debug-capture writer -- and Tk
    variables freed there raise "main thread is not in main loop"."""
    yield
    import gc
    gc.collect()
