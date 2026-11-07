def print_rectangle():
    col = input("how many column should I print?")
    if not col.isdigit():
        raise SystemExit("Please enter a number")
    col = int(col)
    i = 0
    while i <= 4:
        j = 1
        while j <= col:
            print("O", end="")
            j += 1
        i += 1
        print()


if __name__ == "__main__":
    print_rectangle()
