from __future__ import annotations


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. '4.1 GB')."""
    if num_bytes < 0:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
