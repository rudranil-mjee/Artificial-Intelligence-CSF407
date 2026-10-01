"""
Simple Logical Planning Agent
==============================

Implements a STRIPS-style planner:
- A state is a set of ground logical propositions (strings), e.g. {"At(Robot,A)"}.
- An action has:
    name
    pos_preconditions  : propositions that must be TRUE in the current state
    neg_preconditions  : propositions that must be FALSE (absent) in the current state
    pos_effects        : propositions to ADD to the state after applying the action
    neg_effects        : propositions to REMOVE from the state after applying the action
- An action is applicable in state S iff:
    pos_preconditions ⊆ S   AND   neg_preconditions ∩ S = ∅
- Applying an action:
    S' = (S - neg_effects) ∪ pos_effects
- Search: breadth-first search (BFS) over the state space, guaranteeing the
  shortest plan (fewest actions) is found, since all actions are treated as
  unit cost.

Assumptions:
1. States are finite sets of ground propositions; there is no first-order
   quantification or variable binding at plan-search time -- actions are
   pre-instantiated (e.g. Move(A,B) is its own action, not a schema).
2. Actions are deterministic: applying an action always produces exactly one
   successor state.
3. The world is fully observable and static except for the robot's own
   actions (classical STRIPS assumption / closed-world assumption: anything
   not in the state set is assumed false).
4. Goal satisfaction only requires the goal propositions to be a subset of
   the final state (extra propositions being true doesn't matter).
5. BFS explores states in order of increasing plan length, so the first goal
   state found yields a shortest plan. This is acceptable for small state
   spaces like this warehouse example; it would need replacing (e.g. with
   A*) for larger problems.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple


State = FrozenSet[str]


@dataclass(frozen=True)
class Action:
    name: str
    pos_preconditions: FrozenSet[str] = field(default_factory=frozenset)
    neg_preconditions: FrozenSet[str] = field(default_factory=frozenset)
    pos_effects: FrozenSet[str] = field(default_factory=frozenset)
    neg_effects: FrozenSet[str] = field(default_factory=frozenset)

    def is_applicable(self, state: State) -> bool:
        """An action is applicable iff all positive preconditions hold in
        the state and none of the negative preconditions hold in the state."""
        return self.pos_preconditions.issubset(state) and \
               self.neg_preconditions.isdisjoint(state)

    def apply(self, state: State) -> State:
        """Apply the action: remove negative effects, then add positive effects."""
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} is not applicable in state {state}")
        new_state = (state - self.neg_effects) | self.pos_effects
        return frozenset(new_state)


def goal_satisfied(state: State, goal: FrozenSet[str]) -> bool:
    return goal.issubset(state)


def bfs_plan(
    initial_state: State,
    actions: List[Action],
    goal: FrozenSet[str],
) -> Optional[List[Action]]:
    """
    Breadth-first search over the state space.
    Returns a list of actions (the plan) if a goal state is reachable,
    or None if no plan exists.
    """
    if goal_satisfied(initial_state, goal):
        return []

    # Each queue entry: (state, path_of_actions_taken)
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

    return None  # no plan found


def run_planner(
    initial_state: State,
    actions: List[Action],
    goal: FrozenSet[str],
) -> None:
    """Run the planner and pretty-print the plan and intermediate states."""
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

    print(f"Plan found with {len(plan)} action(s):\n")
    state = initial_state
    for i, action in enumerate(plan, start=1):
        state = action.apply(state)
        print(f"Step {i}: {action.name}")
        print("  Resulting state: " + ", ".join(sorted(state)))
    print()
    print("Final state satisfies goal:", goal_satisfied(state, goal))


# ---------------------------------------------------------------------------
# Warehouse example from the lab (Test A: solvable problem)
# ---------------------------------------------------------------------------

def warehouse_actions() -> List[Action]:
    actions = []

    # Move actions between connected locations: A-B, B-C
    moves = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]
    for src, dst in moves:
        actions.append(
            Action(
                name=f"Move(Robot,{src},{dst})",
                pos_preconditions=frozenset({f"At(Robot,{src})"}),
                neg_preconditions=frozenset(),
                pos_effects=frozenset({f"At(Robot,{dst})"}),
                neg_effects=frozenset({f"At(Robot,{src})"}),
            )
        )

    # PickUp actions at each location
    for loc in ["A", "B", "C"]:
        actions.append(
            Action(
                name=f"PickUp(Package,{loc})",
                pos_preconditions=frozenset({f"At(Robot,{loc})", f"At(Package,{loc})"}),
                neg_preconditions=frozenset(),
                pos_effects=frozenset({"Holding(Package)"}),
                neg_effects=frozenset({f"At(Package,{loc})"}),
            )
        )

    # Drop actions at each location
    for loc in ["A", "B", "C"]:
        actions.append(
            Action(
                name=f"Drop(Package,{loc})",
                pos_preconditions=frozenset({f"At(Robot,{loc})", "Holding(Package)"}),
                neg_preconditions=frozenset(),
                pos_effects=frozenset({f"At(Package,{loc})"}),
                neg_effects=frozenset({"Holding(Package)"}),
            )
        )

    return actions


if __name__ == "__main__":
    initial_state: State = frozenset({"At(Robot,A)", "At(Package,A)"})
    goal: FrozenSet[str] = frozenset({"At(Package,C)"})
    actions = warehouse_actions()

    print("=" * 60)
    print("TEST A: Solvable warehouse problem")
    print("=" * 60)
    run_planner(initial_state, actions, goal)

    print()
    print("=" * 60)
    print("TEST B: Impossible problem (PickUp action removed)")
    print("=" * 60)
    actions_no_pickup = [a for a in actions if not a.name.startswith("PickUp")]
    run_planner(initial_state, actions_no_pickup, goal)

    print()
    print("=" * 60)
    print("TEST C: Irrelevant actions (robot alone reaching C != goal)")
    print("=" * 60)
    # Same action set as Test A, but goal only cares about the package,
    # so the planner must not stop just because the robot reaches C.
    run_planner(initial_state, actions, goal)
