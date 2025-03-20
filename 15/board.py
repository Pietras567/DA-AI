class Board:
    def __init__(self, state):
        self.state = state
        self.rows = len(state)
        self.cols = len(state[0])
        self.goal_state = self.get_solved_state()

    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.state])

    def print_state(state):
        for row in state:
            print('\t'.join(map(str, row)))

    def __len__(self):
        return len(self.state)

    def getBlankPosition(self):
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    return i, j

    def move(self, direction):
        moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        row, col = self.getBlankPosition()
        d_row, d_col = moves[direction]
        new_row, new_col = row + d_row, col + d_col

        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            new_state = [row[:] for row in self.state]
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            return new_state
        return self.state

    def is_solved(self):
        return self.state == self.goal_state

    def get_solved_state(self):
        goal = [[i * self.cols + j + 1 for j in range(self.cols)] for i in range(self.rows)]
        goal[self.rows - 1][self.cols - 1] = 0
        return goal

    def get_state(self):
        return self.state


