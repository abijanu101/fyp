"""Wall-clock parsing shared by frame filenames ('10_40_43_442466') and labels ('10:40:43:047895')."""

import re

_CLOCK = re.compile(r"(\d{1,2})[_:](\d{2})[_:](\d{2})[_:](\d+)")


def parse_clock(text: str) -> float:
    """Return seconds since midnight. The fractional part may have any number of digits."""
    m = _CLOCK.fullmatch(text)
    if not m:
        raise ValueError(f"Not a HH?MM?SS?frac timestamp: {text!r}")
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)
