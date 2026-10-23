"""winkeys' choice of screens from mss's monitor list. Run: python -m pytest"""
import winkeys

ALL = {"left": -1920, "top": 0, "width": 4480, "height": 1440}
SIDE = {"left": -1920, "top": 360, "width": 1920, "height": 1080}
GAME = {"left": 0, "top": 0, "width": 2560, "height": 1440}


def test_the_primary_is_found_wherever_mss_lists_it():
    # 2026-10-11: mss listed the 1920x1080 side screen as monitor 1, and every read missed.
    assert winkeys.primary_monitor([ALL, SIDE, GAME]) == GAME
    assert winkeys.primary_monitor([ALL, GAME, SIDE]) == GAME


def test_the_other_screen_is_not_the_primary():
    assert winkeys.side_monitor([ALL, SIDE, GAME]) == SIDE
    assert winkeys.side_monitor([ALL, GAME, SIDE]) == SIDE


def test_one_screen_is_both():
    one = {"left": 0, "top": 0, "width": 2560, "height": 1440}
    assert winkeys.primary_monitor([one, one]) == one
    assert winkeys.side_monitor([one, one]) == one


def test_a_switched_off_key_is_never_down(monkeypatch):
    # live 2026-10-14: GetAsyncKeyState(0) says DOWN on the USER's PC, so the unset --aim-key (0) held
    # "hold to aim" on and the target followed the cursor over the map
    monkeypatch.setattr(winkeys.user32, "GetAsyncKeyState", lambda vk: 0x8000)
    k = winkeys.Key(0)
    assert not k.is_down() and not k.pressed()
    assert winkeys.Key(0x75).is_down()                    # a real key still reads
