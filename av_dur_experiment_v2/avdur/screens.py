"""Which screen the experiment window opens on.

PsychoPy numbers screens in pyglet's order, which need not match Windows' display numbers.
A profile can therefore name the participant's screen by its resolution (`screen_px`):
the lab CRT is set to 800 x 800, which no ordinary monitor uses, so it is found wherever it
is in the list. Order of choice: an explicit screen number (launcher menu or --screen), then
the one screen with the profile's resolution, then the profile's screen number.
"""

from __future__ import annotations


def list_screens():
    """[{index, width, height, x, y}] in PsychoPy's numbering (empty if no display is available)."""
    try:
        import pyglet
        try:
            display = pyglet.canvas.get_display()        # pyglet 1.5 (PsychoPy 2024.2)
        except AttributeError:
            display = pyglet.display.get_display()       # pyglet 2
        screens = display.get_screens()
    except Exception:
        return []
    return [dict(index=i, width=s.width, height=s.height, x=s.x, y=s.y) for i, s in enumerate(screens)]


def describe(screen):
    return f"screen {screen['index']}: {screen['width']} x {screen['height']} at ({screen['x']}, {screen['y']})"


def choose(profile, override=None, screens=None):
    """(screen index, how it was chosen)."""
    screens = list_screens() if screens is None else screens
    if override is not None:
        if screens and not any(s["index"] == override for s in screens):
            raise ValueError(f"There is no screen {override}. Screens: " + "; ".join(map(describe, screens)))
        return int(override), "chosen in the launcher / --screen"
    wanted = profile.get("screen_px")
    if wanted and screens:
        matches = [s for s in screens if (s["width"], s["height"]) == tuple(wanted)]
        if len(matches) == 1:
            return matches[0]["index"], f"the only {wanted[0]} x {wanted[1]} screen"
        if not matches:
            raise RuntimeError(
                f"No screen is set to {wanted[0]} x {wanted[1]}, the participant's display in this profile. "
                "Screens: " + "; ".join(map(describe, screens)) +
                ". Set the participant's monitor to that resolution, or choose the screen in the launcher.")
    return int(profile.get("screen", 0)), "the profile's screen number"
