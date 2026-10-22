def pack():
    """26 equal circles on a 6 x 5 grid, leaving the last four cells empty."""
    circles = []
    for i in range(26):
        row, col = divmod(i, 6)
        circles.append(((col + 0.5) / 6, (row + 0.5) / 5, 0.05))
    return circles
