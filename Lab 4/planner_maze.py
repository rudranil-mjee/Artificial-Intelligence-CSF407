"""
Simple Logical Planning Agent — Warehouse Maze Edition
========================================================

This extends the original STRIPS-style planner so that it operates over the
warehouse maze from the lab handout, with locations named as (x, y)
coordinates instead of letters A, B, C.

Coordinate system
------------------
The maze is read from an ASCII map. The BOTTOM-LEFT free cell is (1, 1).
x increases to the right, y increases upward (so the top row of the ASCII
map has the largest y value). This means the ASCII map must be flipped
vertically when converting row/column positions into (x, y) coordinates.

State representation
---------------------
A state is still a frozenset of ground propositions, e.g.:
    "At(Robot,10,4)"
    "At(Package,10,4)"
    "Holding(Package)"

Action representation
----------------------
Each action still has:
    name
    pos_preconditions  : propositions that must be TRUE in the current state
    neg_preconditions  : propositions that must be FALSE in the current state
    pos_effects        : propositions ADDED after applying the action
    neg_effects        : propositions REMOVED after applying the action

An action is applicable in state S iff:
    pos_preconditions ⊆ S   AND   neg_preconditions ∩ S = ∅
Applying an action:
    S' = (S - neg_effects) ∪ pos_effects

Actions in this warehouse:
  - Move(Robot,(x1,y1),(x2,y2))   for every pair of adjacent free cells
  - PickUp(Package,(x,y))         at any free cell
  - Drop(Package,(x,y))           at any free cell

Search
------
Breadth-first search (BFS) over the state space. Because every action has
unit cost, BFS finds a plan with the fewest actions, i.e. a SHORTEST plan.
This includes both the robot's shortest walk through the maze AND the
optimal point in that walk to detour for the package (if a detour is ever
needed) -- the search is over the full (robot position, package position,
holding?) state space, not just over grid cells.

Assumptions
-----------
1. Locations are grid cells; only orthogonal moves (Up/Down/Left/Right) are
   allowed, matching the lab's maze scenario.
2. Actions are pre-grounded per coordinate pair (Move((x1,y1),(x2,y2)) is
   its own action instance), not a first-order schema evaluated at search
   time.
3. Closed-world assumption: anything not explicitly in the state set is
   assumed false.
4. Actions are deterministic: one action always produces exactly one
   successor state.
5. Goal satisfaction requires the goal propositions to be a SUBSET of the
   final state; extra true propositions do not block success.
6. All actions have unit cost, so BFS's first solution found is a shortest
   plan. This keeps the algorithm identical to the earlier lab's classical
   planner; it would need replacing with A*/UCS if action costs varied.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

State = FrozenSet[str]
Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Action representation (unchanged in structure from the original planner)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    name: str
    pos_preconditions: FrozenSet[str] = field(default_factory=frozenset)
    neg_preconditions: FrozenSet[str] = field(default_factory=frozenset)
    pos_effects: FrozenSet[str] = field(default_factory=frozenset)
    neg_effects: FrozenSet[str] = field(default_factory=frozenset)

    def is_applicable(self, state: State) -> bool:
        return self.pos_preconditions.issubset(state) and \
               self.neg_preconditions.isdisjoint(state)

    def apply(self, state: State) -> State:
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} is not applicable in state {state}")
        return frozenset((state - self.neg_effects) | self.pos_effects)


def goal_satisfied(state: State, goal: FrozenSet[str]) -> bool:
    return goal.issubset(state)


def bfs_plan(
    initial_state: State,
    actions: List[Action],
    goal: FrozenSet[str],
) -> Optional[List[Action]]:
    """Breadth-first search over the state space; returns a shortest plan
    (fewest actions) or None if no plan exists."""
    if goal_satisfied(initial_state, goal):
        return []

    frontier = deque()
    frontier.append((initial_state, []))
    visited = {initial_state}

    while frontier:
        state, path = frontier.popleft()

        for action in actions:
            if not action.is_applicable(state):
                continue

            next_state = action.apply(state)
            if next_state in visited:
                continue

            new_path = path + [action]

            if goal_satisfied(next_state, goal):
                return new_path

            visited.add(next_state)
            frontier.append((next_state, new_path))

    return None


def run_planner(initial_state: State, actions: List[Action], goal: FrozenSet[str]) -> None:
    print("Initial state:")
    print("  " + ", ".join(sorted(initial_state)))
    print("Goal:")
    print("  " + ", ".join(sorted(goal)))
    print()

    plan = bfs_plan(initial_state, actions, goal)

    if plan is None:
        print("No plan found")
        return

    if len(plan) == 0:
        print("Goal already satisfied in the initial state.")
        return

    print(f"Plan found with {len(plan)} action(s) (shortest possible):\n")
    state = initial_state
    for i, action in enumerate(plan, start=1):
        state = action.apply(state)
        print(f"Step {i}: {action.name}")
        print("  Resulting state: " + ", ".join(sorted(state)))
    print()
    print("Final state satisfies goal:", goal_satisfied(state, goal))


# ---------------------------------------------------------------------------
# Maze parsing: convert the ASCII map into coordinate-based free cells.
# (1, 1) is the BOTTOM-LEFT free cell; x -> right, y -> up.
# ---------------------------------------------------------------------------

MAZE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

# Where to place the package: a free cell that lies on the shortest S->G
# path, chosen so the robot naturally passes through it (no forced detour
# is required, though the planner would still find the optimal route even
# if a detour were needed).
PACKAGE_COORD: Coord = (10, 4)


def parse_maze(maze: List[str]) -> Tuple[Dict[Coord, str], Coord, Coord]:
    """Returns (free_cells: coord -> symbol, start_coord, goal_coord)."""
    n_rows = len(maze)
    free_cells: Dict[Coord, str] = {}
    start = goal = None

    for row_idx, line in enumerate(maze):
        for col_idx, ch in enumerate(line):
            if ch == "#":
                continue
            x = col_idx + 1
            y = (n_rows - 1 - row_idx) + 1
            free_cells[(x, y)] = ch
            if ch == "S":
                start = (x, y)
            elif ch == "G":
                goal = (x, y)

    if start is None or goal is None:
        raise ValueError("Maze must contain exactly one 'S' and one 'G'.")

    return free_cells, start, goal


def adjacent_free_pairs(free_cells: Dict[Coord, str]) -> Set[Tuple[Coord, Coord]]:
    """All ordered pairs (a, b) of orthogonally-adjacent free cells."""
    pairs: Set[Tuple[Coord, Coord]] = set()
    for (x, y) in free_cells:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in free_cells:
                pairs.add(((x, y), neighbor))
    return pairs


# ---------------------------------------------------------------------------
# Build the planning problem (states, actions) from the maze.
# ---------------------------------------------------------------------------

def build_warehouse_actions(free_cells: Dict[Coord, str]) -> List[Action]:
    actions: List[Action] = []

    # Move actions: one per ordered pair of adjacent free cells.
    for (src, dst) in adjacent_free_pairs(free_cells):
        sx, sy = src
        dx, dy = dst
        actions.append(
            Action(
                name=f"Move(Robot,({sx},{sy}),({dx},{dy}))",
                pos_preconditions=frozenset({f"At(Robot,{sx},{sy})"}),
                neg_preconditions=frozenset(),
                pos_effects=frozenset({f"At(Robot,{dx},{dy})"}),
                neg_effects=frozenset({f"At(Robot,{sx},{sy})"}),
            )
        )

    # PickUp / Drop actions: available at every free cell.
    for (x, y) in free_cells:
        actions.append(
            Action(
                name=f"PickUp(Package,({x},{y}))",
                pos_preconditions=frozenset({f"At(Robot,{x},{y})", f"At(Package,{x},{y})"}),
                neg_preconditions=frozenset(),
                pos_effects=frozenset({"Holding(Package)"}),
                neg_effects=frozenset({f"At(Package,{x},{y})"}),
            )
        )
        actions.append(
            Action(
                name=f"Drop(Package,({x},{y}))",
                pos_preconditions=frozenset({f"At(Robot,{x},{y})", "Holding(Package)"}),
                neg_preconditions=frozenset(),
                pos_effects=frozenset({f"At(Package,{x},{y})"}),
                neg_effects=frozenset({"Holding(Package)"}),
            )
        )

    return actions


def print_maze_with_package(free_cells: Dict[Coord, str], package: Coord) -> None:
    n_rows = len(MAZE)
    n_cols = len(MAZE[0])
    print("Maze (S=start, G=goal, P=package):")
    for row_idx in range(n_rows):
        line_chars = []
        y = (n_rows - 1 - row_idx) + 1
        for col_idx in range(n_cols):
            x = col_idx + 1
            coord = (x, y)
            if coord not in free_cells:
                line_chars.append("#")
            elif coord == package and free_cells[coord] not in ("S", "G"):
                line_chars.append("P")
            else:
                line_chars.append(free_cells[coord])
        print("".join(line_chars))
    print()


if __name__ == "__main__":
    free_cells, start_coord, goal_coord = parse_maze(MAZE)

    print_maze_with_package(free_cells, PACKAGE_COORD)
    print(f"Start (Robot) coordinate : {start_coord}")
    print(f"Goal coordinate          : {goal_coord}")
    print(f"Package coordinate       : {PACKAGE_COORD}")
    print()

    actions = build_warehouse_actions(free_cells)

    sx, sy = start_coord
    px, py = PACKAGE_COORD
    gx, gy = goal_coord

    initial_state: State = frozenset({
        f"At(Robot,{sx},{sy})",
        f"At(Package,{px},{py})",
    })
    goal: FrozenSet[str] = frozenset({f"At(Package,{gx},{gy})"})

    print("=" * 70)
    print("Deliver the package from its location to the goal cell")
    print("(shortest possible plan, via BFS over the full planning state space)")
    print("=" * 70)
    run_planner(initial_state, actions, goal)
