# references
TWOK_RESOLUTION = (2560, 1440)
TWOK_TIMER_COORDS = (70, 50, 170, 100)


def get_monitor(monitors: list[dict], index: int | None = None):
    if not index:
        for monitor in monitors:
            if monitor["left"] == 0 and monitor["top"] == 0 and monitor != monitors[0]:
                return monitor
    try:
        monitor = monitors[index]
    except IndexError:
        return monitors[1]
    return monitor


def get_timer_coords(monitor):
    width, height = monitor["width"], monitor["height"]
    base_width, base_height = TWOK_RESOLUTION
    scale = min(width / base_width, height / base_height)
    return tuple(int(coord * scale) for coord in TWOK_TIMER_COORDS)
