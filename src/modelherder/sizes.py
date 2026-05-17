def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable 2^10 string (e.g. '4.1 GiB')."""
    if num_bytes < 0:
        return "?"
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            # We don't need to express fractions of bytes:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
