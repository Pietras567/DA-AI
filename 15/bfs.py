from board import Board
import sys
from queue import Queue
sys.setrecursionlimit(1000000000)


def bfs(plansza, param):
    print("start")
    znaki = []
    plansze = []
    queue = Queue()

    num_states = 1
    num_processed = 0
    max_depth = 0

    queue.put((plansza, "", 0))
    plansze.append(plansza.get_state())


    for znak in param:
        znaki.append(znak)


    if plansza.is_solved():
        #print("Ended")
        return "", num_states, num_processed, max_depth
    else:
        while not queue.empty():
            curr, path, depth = queue.get()
            if curr.is_solved():
                #print(curr)
                return path, num_states, num_processed, max_depth
            num_processed += 1
            for x in znaki:
                newBoard = Board(curr.move(x))
                #if (newBoard.get_state() in plansze): print("była")
                #if (newBoard.get_state() == curr.get_state()): print("była2")
                if (newBoard.get_state() == curr.get_state()) | (newBoard.get_state() in plansze):
                    #print("Ta sama lub była")
                    continue
                else:
                    plansze.append(newBoard.get_state())
                    num_states += 1
                    new_path = path + x
                    max_depth = max(max_depth, depth)
                    queue.put((newBoard, new_path, depth + 1))
                    #print(newBoard)
        return None, num_states, num_processed, max_depth


