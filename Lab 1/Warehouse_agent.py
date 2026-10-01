from collections import deque

def solve_warehouse_maze(maze):
    """
    Implements a goal-based agent using BFS and visually traces the final path.
    """
    rows = len(maze)
    cols = len(maze[0])
    
    start_pos = None
    goal_pos = None

    for r in range(rows):
        # Pad uneven rows with spaces automatically to prevent crashes
        maze[r] = maze[r].ljust(cols, '.')
        for c in range(cols):
            if maze[r][c] == 'S':
                start_pos = (r, c)
            elif maze[r][c] == 'G':
                goal_pos = (r, c)

    if not start_pos or not goal_pos:
        print("Error: Start (S) or Goal (G) not found.")
        return

    directions = {
        "Up": (-1, 0),
        "Down": (1, 0),
        "Left": (0, -1),
        "Right": (0, 1)
    }

    queue = deque([(start_pos[0], start_pos[1], [])])
    visited = set([start_pos])

    while queue:
        r, c, path = queue.popleft()

        if (r, c) == goal_pos:
            print(f"\nGoal Reached! Shortest path length: {len(path)} steps.")
            print(f"Path sequence: {path}\n")
            
            # --- NEW VISUALIZATION LOGIC ---
            # Convert maze to a mutable list of lists
            visual_maze = [list(row) for row in maze]
            
            # Trace the path from start to goal
            curr_r, curr_c = start_pos
            for move in path:
                dr, dc = directions[move]
                curr_r, curr_c = curr_r + dr, curr_c + dc
                
                # Mark path with '*' but keep 'G' visible
                if visual_maze[curr_r][curr_c] != 'G':
                    visual_maze[curr_r][curr_c] = '*'
            
            print("Visual Map of Path Taken:")
            for row in visual_maze:
                print("".join(row))
            return path

        for move_name, (dr, dc) in directions.items():
            new_r, new_c = r + dr, c + dc

            if 0 <= new_r < rows and 0 <= new_c < cols:
                if maze[new_r][new_c] != '#' and (new_r, new_c) not in visited:
                    visited.add((new_r, new_c))
                    queue.append((new_r, new_c, path + [move_name]))

    print("No collision-free path exists.")
    return None

if __name__ == "__main__":
    warehouse_map = [
        "#####################",
        "#S....#............G#",
        "#.##....##########..#",
        "#....##.............#",
        "#.######.###.#.###..#",
        "#........#..........#",
        "#####################"
    ]
    
    solve_warehouse_maze(warehouse_map)