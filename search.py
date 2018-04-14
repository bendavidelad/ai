"""
In search.py, you will implement generic search algorithms
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()




def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    return search_helper(problem, util.Stack())


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    return search_helper(problem, util.Queue())


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    counter = 0
    fringe = util.PriorityQueue()
    visited_states = set()
    current_state = [problem.get_start_state(), [], 0]
    successors = problem.get_successors(current_state[0])
    for successor in successors:
        fringe.push((counter, [successor, [successor[1]], successor[2]]), successor[2])
        counter += 1

    while len(fringe.heap) != 0:
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[1][0][0]):
            return current_state[1][1]
        else:
            if current_state[1][0][0] not in visited_states:
                visited_states.add(current_state[1][0][0])
                successors = problem.get_successors(current_state[1][0][0])
                for successor in successors:
                    fringe.push((counter, [successor, current_state[1][1] + [successor[1]], current_state[1][2] +
                                           successor[2]]), current_state[1][2] + successor[2])
                    counter += 1
    return []

def search_helper(problem, fringe):
    visited_states = set()
    current_state = [(problem.get_start_state(), -1, 0), []]
    successors = problem.get_successors(current_state[0][0])
    for successor in successors:
        fringe.push([successor, [successor[1]]])

    while len(fringe.list) != 0:
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[0][0]):
            return current_state[1]
        else:
            if current_state[0][0] not in visited_states:
                visited_states.add(current_state[0][0])
                successors = problem.get_successors(current_state[0][0])
                for successor in successors:
                    fringe.push([successor, current_state[1] + [successor[1]]])
    return []

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    counter = 0
    fringe = util.PriorityQueue()
    visited_states = set()
    current_state = [problem.get_start_state(), [], 0]
    successors = problem.get_successors(current_state[0])
    for successor in successors:
        fringe.push((counter, [successor, [successor[1]], successor[2]]), successor[2]
                    + heuristic(successor[0], problem))
        counter += 1

    while len(fringe.heap) != 0:
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[1][0][0]):
            return current_state[1][1]
        else:
            if current_state[1][0][0] not in visited_states:
                visited_states.add(current_state[1][0][0])
                successors = problem.get_successors(current_state[1][0][0])
                for successor in successors:
                    fringe.push((counter, [successor, current_state[1][1] + [successor[1]], current_state[1][2] +
                                           successor[2]]), current_state[1][2] + successor[2] + heuristic(successor[0], problem))
                    counter += 1
    return []


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
