import heapq
from collections import deque
import math

def calculate_heuristic(current, goal, heuristic_type="manhattan"):
    """Calculates the estimated cost from n to the goal: h(n)"""
    r1, c1 = current
    r2, c2 = goal
    
    if heuristic_type == "manhattan":
        return abs(r1 - r2) + abs(c1 - c2)
    elif heuristic_type == "euclidean":
        return math.sqrt((r1 - r2)**2 + (c1 - c2)**2)
    elif heuristic_type == "zero":
        return 0
    elif heuristic_type == "doubled":
        return 2 * (abs(r1 - r2) + abs(c1 - c2))
    return 0

def solve_warehouse(maze, algorithm="A*", heuristic_type="N/A"):
    """
    Unified search agent that returns performance metrics as a dictionary 
    for tabular formatting.
    """
    rows, cols = len(maze), len(maze[0])
    start_pos = goal_pos = None

    # Parse Environment and fix uneven rows automatically
    for r in range(rows):
        maze[r] = maze[r].ljust(cols, '.')
        for c in range(cols):
            if maze[r][c] == 'S': start_pos = (r, c)
            elif maze[r][c] == 'G': goal_pos = (r, c)

    if not start_pos or not goal_pos:
        return {"Algorithm": algorithm, "Heuristic": heuristic_type, "Found": "Error", "Length": "N/A", "Expanded": 0}

    directions = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
    visited = set()
    states_expanded = 0

    if algorithm in ["A*", "Greedy"]:
        frontier = []
        heapq.heappush(frontier, (0, 0, start_pos, []))
    else:
        frontier = deque([(start_pos, [])])

    while frontier:
        if algorithm in ["A*", "Greedy"]:
            f_n, g_n, current, path = heapq.heappop(frontier)
        elif algorithm == "BFS":
            current, path = frontier.popleft() # FIFO
            g_n = len(path)
        elif algorithm == "DFS":
            current, path = frontier.pop() # LIFO
            g_n = len(path)

        if current in visited:
            continue
            
        visited.add(current)
        states_expanded += 1

        # Goal Test
        if current == goal_pos:
            return {
                "Algorithm": algorithm,
                "Heuristic": heuristic_type,
                "Found": "Yes",
                "Length": len(path),
                "Expanded": states_expanded
            }

        # Expand transitions
        r, c = current
        for move_name, (dr, dc) in directions.items():
            new_r, new_c = r + dr, c + dc

            if 0 <= new_r < rows and 0 <= new_c < cols and maze[new_r][new_c] != '#':
                if (new_r, new_c) not in visited:
                    new_path = path + [move_name]
                    new_g_n = g_n + 1

                    if algorithm in ["A*", "Greedy"]:
                        h_n = calculate_heuristic((new_r, new_c), goal_pos, heuristic_type)
                        new_f_n = new_g_n + h_n if algorithm == "A*" else h_n
                        heapq.heappush(frontier, (new_f_n, new_g_n, (new_r, new_c), new_path))
                    else:
                        frontier.append(((new_r, new_c), new_path))

    # If the queue empties and no goal is found
    return {
        "Algorithm": algorithm,
        "Heuristic": heuristic_type,
        "Found": "No",
        "Length": "N/A",
        "Expanded": states_expanded
    }

def print_results_table(results):
    """Formats and prints the list of result dictionaries as an ASCII table."""
    # Define column widths
    col_alg = 12
    col_heu = 12
    col_fnd = 10
    col_len = 12
    col_exp = 16
    
    # Print Table Header
    header = f"| {'Algorithm':<{col_alg}} | {'Heuristic':<{col_heu}} | {'Found?':<{col_fnd}} | {'Path Length':<{col_len}} | {'States Expanded':<{col_exp}} |"
    divider = "-" * len(header)
    
    print("\n" + divider)
    print(header)
    print(divider)
    
    # Print Table Rows
    for res in results:
        print(f"| {res['Algorithm']:<{col_alg}} | {res['Heuristic']:<{col_heu}} | {res['Found']:<{col_fnd}} | {str(res['Length']):<{col_len}} | {str(res['Expanded']):<{col_exp}} |")
    
    print(divider + "\n")

if __name__ == "__main__":
    # The original warehouse map from the PDF
    warehouse_map = [
        "#################",
        "#S....#.........#",
        "#.###.#.#######.#",
        "#...#.#.......#.#",
        "###.#.#######.#.#",
        "#...#.........#.#",
        "#.###########.#.#",
        "#.............#G#",
        "#################"
    ]

    # Run experiments and collect data
    experiments_data = []
    
    # Blind Searches
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="BFS"))
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="DFS"))
    
    # Informed Searches (Task 5 & 6)
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="A*", heuristic_type="manhattan"))
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="A*", heuristic_type="euclidean"))
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="A*", heuristic_type="zero"))
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="A*", heuristic_type="doubled"))
    
    experiments_data.append(solve_warehouse(warehouse_map.copy(), algorithm="Greedy", heuristic_type="manhattan"))

    # Print the final tabular output
    print_results_table(experiments_data)