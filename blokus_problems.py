from board import Board
from search import SearchProblem, ucs
import util
import copy
import numpy as np
from search import *

FILTER_4_CONN = [[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]]

FILTER_8_CONN = [[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]]


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        self.start_point = starting_point
        self.board_final_pos = [(0, 0), (0, self.board.board_w -1)
                                , (self.board.board_h - 1, 0),
                                (self.board.board_h - 1, self.board.board_w - 1)]
        self.board_final_pos.remove(self.start_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        state0 = state.state
        if state0[0, 0] == 0 and state0[len(state0)-1, 0] == 0 and state0[0, len(state0[0])-1] == 0 and \
                                                                        state0[len(state0)-1, len(state0[0])-1] == 0:
            return True
        else:
            return False

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            total_cost += action.piece.get_num_tiles()
        return total_cost


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    boundary = findBoundary(state.state)
    return max(distManattanFromBounMAX(problem.board_final_pos[0], boundary),
               distManattanFromBounMAX(problem.board_final_pos[1], boundary),
               distManattanFromBounMAX(problem.board_final_pos[2], boundary))

def manhattanDistance(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

def distManattanFromBounMAX(x1, bound): # MAXIMUN from boundary!
    return max(manhattanDistance(x1, b) for b in bound)

class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0
        self.start_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        state0 = state.state
        for i, j in self.targets:
            if(state0[i, j] != 0):
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            total_cost += action.piece.get_num_tiles()
        return total_cost #How much remain not empty is the cost

def conv2d(img, i, j, filter):
    '''
    Get convolution value for pixel in image
    :param img: gray scale image to be convolved
    :param i: x coordinate in the image
    :param j: y coordinate in the image
    :param filter: filter to be apply
    :return: convolution value of the filter and image
             for pixel in image
    '''
    sumC = 0
    numRowF, numberColF = len(filter), len(filter[0])
    for y in range(0, numberColF):
        for x in range(0, numRowF):
           sumC += filter [x][y] * img[i + ((numRowF - 1)//2) - x][j + (numberColF - 1)// 2 - y]
    return sumC

def padding(im, r, c): # padding for the convolution
    '''
    Padding the image in zeros around it
    :param im: gray scale image to be padding
    :param r: number of padding in width
    :param c: number of padding in height
    :return: padded image with zeros around it
    '''
    numCol, numRow = len(im[0]), len(im)
    for i in range(numRow):
        for t in range(r):
            im[i].insert(0, 0)
            im[i].append(0)
    im.insert(0, [0] * (numCol + 2*r + c))
    im.insert(-1, [0] * (numCol + 2* r + c))
    return im

def zeroAll(im):
    '''
    Assign zero value to all the pixel with positive value (not belong to the hole)
    :param im: gray scale image
    :return: image with zero value in pixels that not belong
            to the hole
    '''
    for k in range(len(im)):
        for t in range(len(im[0])):
            if im[k][t] >= 0:
                im[k][t] = 0
    return im

def findBoundary(img, connectivity=4):
    '''
    Finding hole's boundary coordinates
    :param img: gray scale image with hole
    :param connectivity: pixel's connectivity- can be 4 or 8
    :return: boundary of the hole
    '''
    Boundary = []
    if (connectivity == 4):
        filterConv = FILTER_4_CONN
    else:
        filterConv = FILTER_8_CONN
    img2 = padding(copy.deepcopy(img).tolist(), len(filterConv[0])//2, len(filterConv)//2)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if (conv2d(img2, i+len(filterConv[0])//2, j+len(filterConv)//2, filterConv) < 0 and img[i][j] >= 0):
                Boundary.append([i,j])
    return Boundary

def blokus_cover_heuristic(state, problem):
    dist = []
    boundary = findBoundary(state)
    for targ in problem.targets:
        pass

def EuclidieanDistance(t1):
    return np.sqrt((t1[0])^2 + (t1[1]^2)^2)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        self.targets = targets.copy()
        self.current_t = None
        self.start_point = starting_point
        self.currBou = []

    def manhattanDistance(self, x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

    def manhattanDistance2(self, xy1):
        "Returns the Manhattan distance between points xy1 and xy2"
        return abs(xy1[0] - self.start_point[0]) + abs(xy1[1] - self.start_point[1])

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        state0 = state.state
        if(state0[self.current_t[0], self.current_t[1]] != 0):
            return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            total_cost += action.piece.get_num_tiles()
        return total_cost #How much remain not empty is the cost

    def conv2d(self, img, i, j, filter):
        '''
        Get convolution value for pixel in image
        :param img: gray scale image to be convolved
        :param i: x coordinate in the image
        :param j: y coordinate in the image
        :param filter: filter to be apply
        :return: convolution value of the filter and image
                 for pixel in image
        '''
        sumC = 0
        numRowF, numberColF = len(filter), len(filter[0])
        for y in range(0, numberColF):
            for x in range(0, numRowF):
               sumC += filter [x][y] * img[i + ((numRowF - 1)//2) - x][j + (numberColF - 1)// 2 - y]
        return sumC

    def padding(self, im, r, c): # padding for the convolution
        '''
        Padding the image in zeros around it
        :param im: gray scale image to be padding
        :param r: number of padding in width
        :param c: number of padding in height
        :return: padded image with zeros around it
        '''
        numCol, numRow = len(im[0]), len(im)
        for i in range(numRow):
            for t in range(r):
                im[i].insert(0, 0)
                im[i].append(0)
        im.insert(0, [0] * (numCol + 2*r + c))
        im.insert(-1, [0] * (numCol + 2* r + c))
        return im

    def zeroAll(self, im):
        '''
        Assign zero value to all the pixel with positive value (not belong to the hole)
        :param im: gray scale image
        :return: image with zero value in pixels that not belong
                to the hole
        '''
        for k in range(len(im)):
            for t in range(len(im[0])):
                if im[k][t] >= 0:
                    im[k][t] = 0
        return im

    def findBoundary(self, img, connectivity=4):
        '''
        Finding hole's boundary coordinates
        :param img: gray scale image with hole
        :param connectivity: pixel's connectivity- can be 4 or 8
        :return: boundary of the hole
        '''
        Boundary = []
        if (connectivity == 4):
            filterConv = FILTER_4_CONN
        else:
            filterConv = FILTER_8_CONN
        img2 = self.padding(copy.deepcopy(img).tolist(), len(filterConv[0])//2, len(filterConv)//2)
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (self.conv2d(img2, i+len(filterConv[0])//2, j+len(filterConv)//2, filterConv) < 0 and img[i][j] >= 0):
                    Boundary.append([i,j])
        return Boundary

    def distManattanFromBoun(self, x1):
        return min(self.manhattanDistance(x1, b) for b in self.currBou)

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        current_state = self.board.__copy__()
        backtrace = []
        targetQueue = util.PriorityQueueWithFunction(self.manhattanDistance2)
        for t in self.targets:
            targetQueue.push(t)
        while(len(targetQueue.heap) != 0):
            t = targetQueue.pop()
            self.current_t = t
            actions = a_star_search(self)
            for mov in actions:
                self.board.add_move(0, mov)
            backtrace += actions
            self.currBou = self.findBoundary(self.board.state, 8)
            print(self.board)
            print(self.currBou)
            targetQueueNew = util.PriorityQueueWithFunction(self.distManattanFromBoun)
            for t in targetQueue.heap:
                targetQueueNew.push(t[1])
            targetQueue = targetQueueNew
            print(targetQueueNew.heap)
        self.board = current_state
        return backtrace

class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


