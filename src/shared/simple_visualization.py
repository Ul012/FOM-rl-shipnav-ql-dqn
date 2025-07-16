def print_wide_grid():
    grid = [
        ['A', ' ', '#', ' ', 'Z'],
        [' ', '#', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', '#', ' '],
        [' ', ' ', ' ', ' ', ' ']
    ]

    # Breitere Zellen erzeugen (5 Zeichen pro Feld)
    cell_width = 5
    horizontal_line = "+" + "+".join(["-" * cell_width] * 5) + "+"

    for row in grid:
        print(horizontal_line)
        row_str = ""
        for cell in row:
            content = f" {cell} " if cell != ' ' else "   "
            row_str += f"|{content.center(cell_width)}"
        row_str += "|"
        print(row_str)
    print(horizontal_line)

if __name__ == "__main__":
    print_wide_grid()
