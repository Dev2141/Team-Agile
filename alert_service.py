"""
alert_service.py — ThreatSense AI-DVR
Non-blocking helper that fires a desktop notification + audio beep on threat detection.

Dependencies:
    pip install plyer
    (winsound is built-in on Windows; on Linux/macOS uses `subprocess` to play a beep)
"""

import threading
import platform
import os


# ──────────────────────────────────────────────
# Cross-platform audio beep
# ──────────────────────────────────────────────
def _play_beep() -> None:
    """Play a short attention-grabbing beep (non-blocking, called in a thread)."""
    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            # Frequency 1000 Hz, duration 400 ms
            winsound.Beep(1000, 400)
        elif system == "Darwin":
            # macOS — use built-in afplay on the system alert sound
            os.system("afplay /System/Library/Sounds/Sosumi.aiff")
        else:
            # Linux — try paplay / aplay fallback, then bell char
            result = os.system("paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null")
            if result != 0:
                print("\a", end="", flush=True)   # ASCII terminal bell
    except Exception as exc:
        # Never crash the main pipeline over a missing sound
        print(f"[alert_service] Audio beep failed (non-critical): {exc}")


# ──────────────────────────────────────────────
# Desktop notification
# ──────────────────────────────────────────────
def _push_notification(camera_name: str, reasoning: str) -> None:
    """Fire a native desktop notification via plyer."""
    try:
        from plyer import notification

        notification.notify(
            title=f"Alert: {camera_name}",
            message=reasoning[:256] if reasoning else "Suspicious activity identified.",
            app_name="ThreatSense System",
            # timeout in seconds (supported on most platforms)
            timeout=8,
        )
    except Exception as exc:
        print(f"[alert_service] Desktop notification failed (non-critical): {exc}")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def trigger_alert(camera_name: str, reasoning: str, is_threat: bool = True) -> None:
    """
    Non-blocking entry point.  Dispatch the beep and notification on
    background daemon threads so the calling AI pipeline is never stalled.

    Args:
        camera_name: Display name of the camera that raised the alarm.
        reasoning:   Short natural-language explanation from Ollama.
        is_threat:   If True, play alarm beep. If False (e.g. SAFE verdict), notification only.
    """
    notify_thread = threading.Thread(
        target=_push_notification,
        args=(camera_name, reasoning),
        daemon=True,
    )
    notify_thread.start()

    if is_threat:
        beep_thread = threading.Thread(target=_play_beep, daemon=True)
        beep_thread.start()

    print(
        f"[INFO] alert_service: Alert dispatched | camera='{camera_name}' | "
        f"reasoning='{reasoning[:80]}...'"
    )


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    trigger_alert(
        camera_name="CAM-01 (Front Gate)",
        reasoning="Unidentified individual loitering near the entrance for >120 seconds.",
    )
    import time
    time.sleep(3)   # Keep process alive long enough for daemon threads to finish
    print("[INFO] alert_service: Smoke-test complete.")
