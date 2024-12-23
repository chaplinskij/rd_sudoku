from collections import Counter

class SudokuSolver:
    @classmethod
    def solve(cls, input: list):
        if not cls.check_input(input):
            return None
        original = input[:]
        if cls.backtrack(input, 0):
            return [input[i] if original[i] == 0 else 0 for i in range(81)]
        else:
            return None

    @classmethod
    def is_valid(cls, grid, num, row, col):
        # Проверяем строку
        for c in range(9):
            if grid[row * 9 + c] == num:
                return False

        # Проверяем столбец
        for r in range(9):
            if grid[r * 9 + col] == num:
                return False

        # Проверяем 3x3 квадрат
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if grid[r * 9 + c] == num:
                    return False

        return True

    @classmethod
    def backtrack(cls, grid, index):
        if index == 81:
            return True  # Все клетки заполнены

        if grid[index] != 0:
            return cls.backtrack(grid, index + 1)  # Переходим к следующей клетке

        row, col = divmod(index, 9)
        for num in range(1, 10):
            if cls.is_valid(grid, num, row, col):
                grid[index] = num
                if cls.backtrack(grid, index + 1):
                    return True
                grid[index] = 0  # Отмена выбора

        return False

    @classmethod
    def check_input(cls, input: list) -> bool:
        if len(input) != 81:
            return False
        input = [num for num in input if num != 0]
        if len(input) < 17:
            return False
        counter = Counter(input)
        if max(counter.values()) > 9:
            return False

        return True
