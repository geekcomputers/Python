from queue import PriorityQueue


class PuzzleState:
    def __init__(self, board, goal, moves=0, previous=None):
        self.board = board
        self.goal = goal
        self.moves = moves
        self.previous = previous

    def __lt__(self, other):
        return self.priority() < other.priority()

    def priority(self):
        return self.moves + self.manhattan()

    def manhattan(self):
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    x, y = divmod(self.board[i][j] - 1, 3)
                    distance += abs(x - i) + abs(y - j)
        return distance

    def is_goal(self):
        return self.board == self.goal

    def neighbors(self):
        neighbors = []
        x, y = next((i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(
                    PuzzleState(new_board, self.goal, self.moves + 1, self)
                )

        return neighbors


def solve_puzzle(initial_board, goal_board):
    initial_state = PuzzleState(initial_board, goal_board)
    frontier = PriorityQueue()
    frontier.put(initial_state)
    explored = set()

    while not frontier.empty():
        current_state = frontier.get()

        if current_state.is_goal():
            return current_state

        explored.add(tuple(map(tuple, current_state.board)))

        for neighbor in current_state.neighbors():
            if tuple(map(tuple, neighbor.board)) not in explored:
                frontier.put(neighbor)

    return None


def print_solution(solution):
    steps = []
    while solution:
        steps.append(solution.board)
        solution = solution.previous
    steps.reverse()

    for step in steps:
        for row in step:
            print(" ".join(map(str, row)))
        print()


# Example usage
initial_board = [[1, 2, 3], [4, 0, 5], [7, 8, 6]]

goal_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

solution = solve_puzzle(initial_board, goal_board)
if solution:
    print("Solution found:")
    print_solution(solution)
else:
    print("No solution found.")
