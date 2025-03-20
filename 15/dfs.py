from board import Board
import sys
sys.setrecursionlimit(1000000000)


def dfs(plansza, param):
    print("Started")
    znaki = []
    plansze = {tuple(map(tuple, plansza.get_state())): 0}
    limit = 20  # Limit głębokości do zdefiniowania
    num_states = 1
    num_processed = 0
    max_depth = 0

    #print("\n" + plansza.__str__() + "\n")

    for znak in param:
        znaki.append(znak)

    def dfs_recursion(curr, path, depth):
        nonlocal num_states, num_processed, max_depth
        curr_board = Board(curr)

        if curr_board.is_solved():
            #print("Ended")
            #print("\n"+curr_board.__str__()+"\n")
            return path, num_states, num_processed, max_depth

        num_processed += 1
        #print("Glebokosc: " + str(depth))
        if depth >= limit:
            return None

        for direction in znaki:
            new_board = Board(curr_board.move(direction))
            if new_board.get_state() != curr_board.get_state():
                new_state = new_board.get_state()

                #if tuple(map(tuple, new_state)) not in plansze: print("Nie Była")
                #if tuple(map(tuple, new_state)) in plansze:
                #    if plansze[tuple(map(tuple, new_state))] > depth: print("Była")

                if tuple(map(tuple, new_state)) not in plansze or plansze[tuple(map(tuple, new_state))] > depth:
                    #print("\nPrzetwarzane\n\n" + str(new_state) + "\n")
                    #print("\nDługość: " + str(len(plansze)))
                    plansze[tuple(map(tuple, new_state))] = depth
                    num_states += 1
                    new_path = path + direction
                    max_depth = max(max_depth, len(new_path))
                    result = dfs_recursion(new_board.get_state(), new_path, depth + 1)
                    if result is not None:
                        return result
        return None

    result = dfs_recursion(plansza.get_state(), "", 0)
    if result is None:
        return -1, num_states, num_processed, max_depth
    else:
        return result




