import time
from bfs import bfs
from dfs import dfs
from board import Board
from astar import *
import sys

# Define the valid strategies and their corresponding parameters
VALID_STRATEGIES = {
    "bfs": ["neighborhood_order"],
    "dfs": ["neighborhood_order"],
    "astr": ["heuristic"],
}

# Define the valid neighborhood orders for BFS and DFS
VALID_NEIGHBORHOOD_ORDERS = ["RLDU", "RLUD", "RDUL", "RDLU", "RUDL", "RULD",
                             "LRDU", "LRUD", "LDUR", "LDRU", "LURD", "LUDR",
                             "DRUL", "DRLU", "DLUR", "DLRU", "DULR", "DURL",
                             "URDL", "URLD", "UDLR", "UDRL", "ULRD", "ULDR"]

# Define the valid heuristics for A*
VALID_HEURISTICS = ["manh", "hamm"]


# Odczyt układu początkowego
def read_initial_state(file_name):
    with open(file_name, 'r') as file:
        next(file)
        state = [list(map(int, line.strip().split())) for line in file]

    board = Board(state)
    return board

# Zapis rozwiązania
def write_solution(file_name, node):
    pass
# Zapis dodatkowych informacji
def write_additional_info(file_name, node, visited_states, processed_states, max_depth, time_taken):
    pass


# Pobranie argumentów wywołania
strategy = sys.argv[1]
param = sys.argv[2]
initial_state_file = sys.argv[3]
solution_file = sys.argv[4]
additional_info_file = sys.argv[5]

# Odczyt układu początkowego
plansza = read_initial_state(initial_state_file)
solution, num_states, num_processed, max_depth = "LLLLLRRRDDU", 0, 0, 0
start_time = None
end_time = None

if strategy not in VALID_STRATEGIES:
    print("Choose valid strategie from:")
    for x in VALID_STRATEGIES:
        print(x)

# Wybór metody przeszukiwania
if strategy == 'bfs':
    start_time = time.time()
    
    if param not in VALID_NEIGHBORHOOD_ORDERS:
        print("Choose valid order from:")
        for x in VALID_NEIGHBORHOOD_ORDERS:
            print(x)


    solution, num_states, num_processed, max_depth = bfs(plansza, param)
    end_time = time.time()

    print("solotion: " + str(solution))
    print("num_states: " + str(num_states))
    print("num_processed: " + str(num_processed))
    print("max_depth: " + str(max_depth))
    print("end_time: " + str(end_time - start_time))
elif strategy == 'dfs':
    start_time = time.time()

    if param not in VALID_NEIGHBORHOOD_ORDERS:
        print("Choose valid order from:")
        for x in VALID_NEIGHBORHOOD_ORDERS:
            print(x)
    solution, num_states, num_processed, max_depth = dfs(plansza, param)


    end_time = time.time()

    print("solotion: " + str(solution))
    print("num_states: " + str(num_states))
    print("num_processed: " + str(num_processed))
    print("max_depth: " + str(max_depth))
    print("end_time: " + str(end_time-start_time))
elif strategy == 'astr':
    start_time = time.time()
    #node, time_taken = astar(initial_state, param)
    if param not in VALID_HEURISTICS:
        print("Choose valid Heuretic from:")
        for x in VALID_HEURISTICS:
            print(x)
    #print("Algorytm A*")
    #print(plansza)

    if param == "manh":
        solution, num_states, num_processed, max_depth = astar(plansza, manhattan_distance)
    elif param == "hamm":
        solution, num_states, num_processed, max_depth = astar(plansza, hamming_distance)

    end_time = time.time()

# Perform the search and measure the execution time

duration = format((end_time - start_time) * 1000, '.3f')

# Write the solution to the output file
with open(solution_file, "w") as f:
    if solution == -1:
        f.write("-1\n")
    else:
        f.write(f"{len(solution)}\n{solution}")

# Write the search stats to the stats file
with open(additional_info_file, "w") as f:
    f.write(f"{len(solution) if solution != -1 else -1}\n"
            f"{num_states}\n"
            f"{num_processed}\n"
            f"{max_depth}\n"
            f"{duration}")
