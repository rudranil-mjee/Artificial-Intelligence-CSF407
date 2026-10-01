from collections import deque

class Action:
    def __init__(self, name, pos_pre, neg_pre, pos_eff, neg_eff):
        self.name = name
        self.pos_pre = set(pos_pre)  # What must be true before the action
        self.neg_pre = set(neg_pre)  
        self.pos_eff = set(pos_eff)  # What becomes true after the action
        self.neg_eff = set(neg_eff)  # What becomes false after the action

    def is_applicable(self, state):
        """An action is applicable if all of its preconditions are satisfied by the current state."""
        if not self.pos_pre.issubset(state):
            return False
        if not self.neg_pre.isdisjoint(state):
            return False
        return True

    def apply(self, state):
        """Removes negative effects and adds positive effects[cite: 3]."""
        new_state = state - self.neg_eff
        new_state = new_state | self.pos_eff
        return new_state

def bfs_planner(initial_state, goal_state, actions):
    """Uses breadth-first search to find a sequence of actions that achieves a specified goal[cite: 3]."""
    queue = deque([(initial_state, [], [initial_state])])
    visited = set([frozenset(initial_state)])

    while queue:
        current_state, plan, history = queue.popleft()

        if goal_state.issubset(current_state):
            print("\nPlan found!")
            print(f"Initial State: {set(history[0])}")
            for i, action in enumerate(plan):
                print(f"Step {i+1}: {action}")
                print(f"  Resulting State: {set(history[i+1])}")
            return plan

        for action in actions:
            if action.is_applicable(current_state):
                new_state = action.apply(current_state)
                if frozenset(new_state) not in visited:
                    visited.add(frozenset(new_state))
                    queue.append((new_state, plan + [action.name], history + [new_state]))

    print("\nNo plan found[cite: 3].")
    return None

def build_warehouse_actions():
    actions = []
    locations = ['A', 'B', 'C']
    connections = [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')] # Robot moves between connected locations[cite: 3]
    
    for loc1, loc2 in connections:
        actions.append(Action(
            name=f"Move({loc1}, {loc2})",
            pos_pre=[f"At(Robot, {loc1})"],
            neg_pre=[],
            pos_eff=[f"At(Robot, {loc2})"],
            neg_eff=[f"At(Robot, {loc1})"]
        ))
        
    for loc in locations:
        actions.append(Action(
            name=f"PickUp(Package, {loc})",
            pos_pre=[f"At(Robot, {loc})", f"At(Package, {loc})"],
            neg_pre=[],
            pos_eff=["Holding(Package)"],
            neg_eff=[f"At(Package, {loc})"]
        ))
        actions.append(Action(
            name=f"Drop(Package, {loc})",
            pos_pre=[f"At(Robot, {loc})", "Holding(Package)"],
            neg_pre=[],
            pos_eff=[f"At(Package, {loc})"],
            neg_eff=["Holding(Package)"]
        ))
    return actions

if __name__ == "__main__":
    all_actions = build_warehouse_actions()
    
    print("=== Test A: Solvable Problem ===")
    initial_A = {"At(Robot, A)", "At(Package, A)"}
    goal_A = {"At(Package, C)"}
    bfs_planner(initial_A, goal_A, all_actions)
    
    print("\n=== Test B: Impossible Problem ===")
    initial_B = {"At(Robot, A)", "At(Package, A)"}
    goal_B = {"At(Package, C)"}
    no_pickup_actions = [a for a in all_actions if "PickUp" not in a.name]
    bfs_planner(initial_B, goal_B, no_pickup_actions)

    print("\n=== Test C: Irrelevant Actions ===")
    initial_C = {"At(Robot, A)", "At(Package, A)"}
    goal_C = {"At(Package, C)"}
    # The planner naturally explores 'Move(A, B)' without holding the package, 
    # but BFS ensures it won't mistake robot reaching C as the package reaching C[cite: 3].
    bfs_planner(initial_C, goal_C, all_actions)