from queue import PriorityQueue
from typing import List, Tuple, Optional, Set


class PuzzleState:
    """Represents a state in 8-puzzle solving with A* algorithm."""

    def __init__(
        self,
        board: List[List[int]],
        goal: List[List[int]],
        moves: int = 0,
        previous: Optional["PuzzleState"] = None,
    ) -> None:
        self.board = board  # Current 3x3 board configuration
        self.goal = goal  # Target 3x3 configuration
        self.moves = moves  # Number of moves taken to reach here
        self.previous = previous  # Previous state in solution path

    def __lt__(self, other: "PuzzleState") -> bool:
        """For PriorityQueue ordering: compare priorities."""
        return self.priority() < other.priority()

    def priority(self) -> int:
        """A* priority: moves + Manhattan distance."""
        return self.moves + self.manhattan()

    def manhattan(self) -> int:
        """Calculate Manhattan distance using actual goal positions."""
        distance = 0
        # Create a lookup table for goal tile positions
        goal_pos = {self.goal[i][j]: (i, j) for i in range(3) for j in range(3)}

        for i in range(3):
            for j in range(3):
                value = self.board[i][j]
                if value != 0:  # skip the empty tile
                    x, y = goal_pos[value]
                    distance += abs(x - i) + abs(y - j)
        return distance


    def is_goal(self) -> bool:
        """Check if current state matches goal."""
        return self.board == self.goal

    def neighbors(self) -> List["PuzzleState"]:
        """Generate all valid neighboring states by moving empty tile (0)."""
        neighbors = []
        x, y = next((i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(
                    PuzzleState(new_board, self.goal, self.moves + 1, self)
                )
        return neighbors


def solve_puzzle(
    initial_board: List[List[int]], goal_board: List[List[int]]
) -> Optional[PuzzleState]:
    """
    Solve 8-puzzle using A* algorithm.

    >>> solve_puzzle([[1,2,3],[4,0,5],[7,8,6]], [[1,2,3],[4,5,6],[7,8,0]]) is not None
    True
    """
    initial = PuzzleState(initial_board, goal_board)
    frontier = PriorityQueue()
    frontier.put(initial)
    explored: Set[Tuple[Tuple[int, ...], ...]] = set()

    while not frontier.empty():
        current = frontier.get()
        if current.is_goal():
            return current
        explored.add(tuple(map(tuple, current.board)))
        for neighbor in current.neighbors():
            if tuple(map(tuple, neighbor.board)) not in explored:
                frontier.put(neighbor)
    return None


def print_solution(solution: Optional[PuzzleState]) -> None:
    """Print step-by-step solution from initial to goal state."""
    if not solution:
        print("No solution found.")
        return
    steps = []
    while solution:
        steps.append(solution.board)
        solution = solution.previous
    for step in reversed(steps):
        for row in step:
            print(" ".join(map(str, row)))
        print()


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
