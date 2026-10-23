"""Windows helpers for artillery.py: polled hotkeys, the foreground window, the cursor,
beeps, and where debug captures go. Nothing here sends input: keys are only read.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import threading
import winsound

DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")

GA_ROOT = 2

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetForegroundWindow.restype = wt.HWND
user32.GetAncestor.argtypes = [wt.HWND, wt.UINT]
user32.GetAncestor.restype = wt.HWND
user32.WindowFromPoint.argtypes = [wt.POINT]
user32.WindowFromPoint.restype = wt.HWND
user32.GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]

NAMED_KEYS = {"MOUSE4": 0x05, "MOUSE5": 0x06, "INSERT": 0x2D, "HOME": 0x24,
              "END": 0x23, "PAUSE": 0x13, "SCROLLLOCK": 0x91, "SPACE": 0x20}


def vk_code(name):
    n = name.upper()
    if n in NAMED_KEYS:
        return NAMED_KEYS[n]
    if n[:1] == "F" and n[1:].isdigit() and 1 <= int(n[1:]) <= 24:
        return 0x6F + int(n[1:])
    if len(n) == 1 and n.isalnum():
        return ord(n)
    raise argparse.ArgumentTypeError(
        f"unknown key {name!r} (use F1-F24, MOUSE4, MOUSE5, a letter/digit, "
        f"or one of {', '.join(NAMED_KEYS)})")


class Key:
    """Edge-triggered hotkey, read by polling -- no hook for Windows to drop."""

    def __init__(self, vk):
        self.vk = vk
        self.was_down = self.is_down()

    def is_down(self):
        # vk 0 is a key that is switched off -- and GetAsyncKeyState(0) says DOWN on this PC (live
        # 2026-10-14: the unset --aim-key held "hold to aim" on, retargeting wherever the map was hovered)
        return bool(self.vk) and bool(user32.GetAsyncKeyState(self.vk) & 0x8000)

    def pressed(self):
        down = self.is_down()
        edge = down and not self.was_down
        self.was_down = down
        return edge


def top_level(hwnd):
    return user32.GetAncestor(hwnd, GA_ROOT) if hwnd else None


def window_title(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value


def window_exe(hwnd):
    """The file name of the program that owns a window ("WardogsClient-Win64-Shipping.exe"), or ""."""
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf, size = ctypes.create_unicode_buffer(512), wt.DWORD(512)
        ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        return buf.value.replace("/", "\\").rsplit("\\", 1)[-1] if ok else ""
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def cursor():
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def cursor_over(hwnd):
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return top_level(user32.WindowFromPoint(pt)) == hwnd


def primary_monitor(monitors):
    """mss's entry for the primary display -- the game's screen. Windows always puts the
    primary at (0, 0), but mss lists screens in EnumDisplayMonitors order, which need not
    start with it: on 2026-10-11 it listed the 1920x1080 side screen as monitor 1, and the
    arrow prompt, the vendor title and the map readout were all read off the wrong screen."""
    return next((m for m in monitors[1:] if m["left"] == 0 and m["top"] == 0), monitors[1])


def side_monitor(monitors):
    """The first screen that is not the primary (the primary itself when there is only one)."""
    game = primary_monitor(monitors)
    return next((m for m in monitors[1:] if m != game), game)


def beep(freq, ms=90):
    threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()


def dpi_aware():
    """Cursor and screen grabs in real pixels, whatever Windows' display scaling says.
    Call before any window or screen grab exists."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
