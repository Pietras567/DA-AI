from queue import PriorityQueue
from board import Board
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        if self.parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def __lt__(self, node):
        return self.state < node.state

    def find_solution(self):
        solution = []
        node = self
        while node:
            solution.append(node.action)
            node = node.parent
        solution = solution[:-1]
        solution.reverse()
        return solution

def hamming_distance(board):
    distance = 0
    goal = board.get_solved_state()
    for i in range(board.rows):
        for j in range(board.cols):
            if board.state[i][j] != goal[i][j] and board.state[i][j] != 0:
                distance += 1
                #print("\nstart")
                #print("Oczekiwana: "+str(goal[i][j]))
                #print("Rzeczywista: " + str(board.state[i][j]))
                #print("Dodaję")
    #print(str(distance) + " koniec")
    return distance

def manhattan_distance(board):
    distance = 0
    for i in range(board.rows):
        for j in range(board.cols):
            if board.state[i][j] != 0:
                goal_row, goal_col = divmod(board.state[i][j]-1, board.cols)
                distance += abs(i - goal_row) + abs(j - goal_col)
    #print(str(distance))
    return distance

def astar(board, heuristic):
    start = Node(board.get_state())
    frontier = PriorityQueue()
    frontier.put((0, start))
    explored = set()
    num_states = 1
    num_processed = 0
    max_depth = 0

    while not frontier.empty():
        _, node = frontier.get()
        board.state = node.state
        explored.add(board.__str__())
        num_processed += 1

        if board.is_solved():
            actions = node.find_solution()
            #print("Koniec: \n"+board.__str__())
            return ''.join(actions), num_states, num_processed, max_depth

        for action in ["U", "D", "L", "R"]:
            child_board = Board(board.move(action))
            child = Node(child_board.get_state(), node, action, node.path_cost + 1)
            #if child_board.__str__() in explored: print("Była")
            #if child_board.__str__() == board.__str__(): print("Była2")
            if (child_board.__str__() not in explored) & (child_board.__str__() != board.__str__()):
                h = heuristic(child_board)
                f = child.path_cost + h
                frontier.put((f, child))
                num_states += 1
                max_depth = max(max_depth, child.depth)

    return None, num_states, num_processed, max_depth