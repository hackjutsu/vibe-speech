from __future__ import annotations

import logging
from typing import Callable, Optional

from pynput import keyboard

logger = logging.getLogger(__name__)


class HotkeyManager:
    """Owns hotkey registration and emits listening state changes."""

    def __init__(self, toggle: str, push_to_talk: bool) -> None:
        self.toggle = toggle
        self.push_to_talk = push_to_talk
        self._listener: Optional[object] = None

    def start(self, on_listen_on: Callable[[], None], on_listen_off: Callable[[], None]) -> None:
        hotkey = self.toggle
        if not hotkey:
            logger.warning("Hotkey toggle not configured; always listening.")
            return
        if self.push_to_talk:
            combo_keys = self._combo_keys(hotkey)
            if not combo_keys:
                logger.warning("Hotkey not understood; push-to-talk disabled.")
                return
            pressed: set[object] = set()

            def on_press(key: object) -> None:
                if key not in combo_keys:
                    return
                pressed.add(key)
                if combo_keys.issubset(pressed):
                    on_listen_on()

            def on_release(key: object) -> None:
                if key in pressed:
                    pressed.remove(key)
                if combo_keys and not combo_keys.issubset(pressed):
                    on_listen_off()
                    pressed.clear()

            listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=False)
            listener.start()
            self._listener = listener
            logger.info("Push-to-talk enabled; hold '%s' to listen.", hotkey)
        else:
            combo = self._to_pynput_combo(hotkey)
            listener = keyboard.GlobalHotKeys({combo: self._toggle_wrapper(on_listen_on, on_listen_off)}, suppress=False)
            listener.start()
            self._listener = listener
            logger.info("Hotkey '%s' registered for start/stop listening.", hotkey)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    def _toggle_wrapper(self, on_on: Callable[[], None], on_off: Callable[[], None]) -> Callable[[], None]:
        state = {"listening": False}

        def _toggle() -> None:
            state["listening"] = not state["listening"]
            if state["listening"]:
                on_on()
            else:
                on_off()

        return _toggle

    def _combo_keys(self, combo: str) -> set[object]:
        mapping = {
            "ctrl": keyboard.Key.ctrl,
            "control": keyboard.Key.ctrl,
            "shift": keyboard.Key.shift,
            "alt": keyboard.Key.alt,
            "option": keyboard.Key.alt,
            "cmd": keyboard.Key.cmd,
            "command": keyboard.Key.cmd,
            "meta": keyboard.Key.cmd,
            "super": keyboard.Key.cmd,
            "space": keyboard.Key.space,
            "enter": keyboard.Key.enter,
            "return": keyboard.Key.enter,
            "tab": keyboard.Key.tab,
            "esc": keyboard.Key.esc,
            "escape": keyboard.Key.esc,
        }
        keys: set[object] = set()
        for part in (p.strip().lower() for p in combo.split("+")):
            key_obj = mapping.get(part)
            if key_obj:
                keys.add(key_obj)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
        return keys

    def _to_pynput_combo(self, combo: str) -> str:
        mapping = {
            "ctrl": "<ctrl>",
            "control": "<ctrl>",
            "shift": "<shift>",
            "alt": "<alt>",
            "option": "<alt>",
            "cmd": "<cmd>",
            "command": "<cmd>",
            "meta": "<cmd>",
            "super": "<cmd>",
            "space": "<space>",
            "enter": "<enter>",
            "return": "<enter>",
            "tab": "<tab>",
            "esc": "<esc>",
            "escape": "<esc>",
        }
        parts = [part.strip().lower() for part in combo.split("+")]
        mapped = [mapping.get(part, part) for part in parts]
        return "+".join(mapped)
