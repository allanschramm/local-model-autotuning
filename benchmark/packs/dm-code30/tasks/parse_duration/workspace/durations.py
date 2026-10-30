UNITS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}


def parse_duration(text: str) -> int:
    """Parse a human duration string into seconds."""
    total = 0
    number = ''
    for char in text:
        if char.isdigit():
            number += char
        elif char in UNITS:
            total += int(number) * UNITS[char]
            number = ''
    return total
