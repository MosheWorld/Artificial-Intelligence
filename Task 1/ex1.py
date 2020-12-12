import math
import heapq

from functools import total_ordering



# Point class will store the point of the grid.
# The points will be transformed into this representation and I will use this for the algorithms.
class Point(object):
    # Defining two variables, one is X and one is Y as cartesian axis.
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # String representation of a point object.
    def __str__(self):
        return f'x:{self.x}, y:{self.y}'

    # Equality implementation for this class.
    def __eq__(self, other):
        if other == None:
            return False
        elif self.x == other.x and self.y == other.y:
            return True
        else:
            return False

    # Lower than implementation for this class.
    def __lt__(self, point):
        return str(self) < str(point)

    # Lower or equals implementation for this class.
    def __le__(self, point):
        return str(self) <= str(point)

    # Greater than implementation for this class.
    def __gt__(self, point):
        return str(self) > str(point)

    # Greater or equals implementation for this class.
    def __ge__(self, point):
        return str(self) >= str(point)

    # Hash implementation for this class.
    def __hash__(self):
        return hash(str(self))



# InputManager class is responsible to read from the file system the input.txt file, validate the arguments
# Then it's validating the inputs that they are valid and transform it to object representation of the input file.
# This class is mainly for comfort reasons, to store everything in an object and not as lists.
class InputManager(object):
    # Defining variables that mirrors exactly the arguments from the input file.
    def __init__(self, path):
        self.path = path
        self.algorithm = None
        self.start_point = None
        self.end_point = None
        self.grid_size = None
        self.grid = None

    # Reads the input.txt from the file system and parse the arguments to variables representation.
    def load(self):
        data = None
        with open(self.path, 'r') as file:
            data = [line.rstrip('\n') for line in file]

        self.algorithm = data[0]
        self.start_point = self.point_str_to_point_object(data[1])
        self.end_point = self.point_str_to_point_object(data[2])
        self.grid_size = int(data[3])
        self.grid = self.read_grid(data[4:])

    # Transforms the string grid to a matrix grid of integers.
    def read_grid(self, grid_str):
        grid = []

        for row in grid_str:
            splitted_row = row.split(',')

            current_new_row = []
            for value in splitted_row:
                current_new_row.append(int(value))

            grid.append(current_new_row)

        return grid

    # Transforms point as string representation to point as Point class representation.
    def point_str_to_point_object(self, point_string):
        splitted_point = point_string.split(',')

        if splitted_point == None or len(splitted_point) != 2:
            raise Exception('Point arguments as string is not valid.')

        return Point(int(splitted_point[0]), int(splitted_point[1]))



# This class is responsible to define the problem I want to solve.
# I define the start point and endpoint, additionally I define how I expand the problem and what it means "cost".
# Also what are my valid actions, where can I go and where I can't go.
class GridSearchProblem(object):
    # Initializes the problem class with the start point, end point, grid and the input manager.
    def __init__(self, input_manager):
        self.input_manager = input_manager
        self.start = input_manager.start_point
        self.goal = input_manager.end_point
        self.grid = input_manager.grid

    # Returns list of actions that can be made from current state, going right? left? and etc.
    def actions(self, state):
        actions = []

        # Right
        if self.is_point_within_borders(state.x, state.y + 1) and self.grid[state.x][state.y + 1] != -1:
            actions.append('R')

        # Right - Down
        if self.is_point_within_borders(state.x + 1, state.y + 1) and self.grid[state.x + 1][state.y + 1] != -1 and self.grid[state.x][state.y + 1] != -1 and self.grid[state.x + 1][state.y] != -1:
            actions.append('RD')

        # Down
        if self.is_point_within_borders(state.x + 1, state.y) and self.grid[state.x + 1][state.y] != -1:
            actions.append('D')

        # Down - Left
        if self.is_point_within_borders(state.x + 1, state.y - 1) and self.grid[state.x + 1][state.y - 1] != -1 and self.grid[state.x + 1][state.y] != -1 and self.grid[state.x][state.y - 1] != -1:
            actions.append('LD')

        # Left
        if self.is_point_within_borders(state.x, state.y - 1) and self.grid[state.x][state.y - 1] != -1:
            actions.append('L')

        # Left - Up
        if self.is_point_within_borders(state.x - 1, state.y - 1) and self.grid[state.x - 1][state.y - 1] != -1 and self.grid[state.x][state.y - 1] != -1 and self.grid[state.x - 1][state.y] != -1:
            actions.append('LU')

        # Up
        if self.is_point_within_borders(state.x - 1, state.y) and self.grid[state.x - 1][state.y] != -1:
            actions.append('U')

        # Up - Right
        if self.is_point_within_borders(state.x - 1, state.y + 1) and self.grid[state.x - 1][state.y + 1] != -1 and self.grid[state.x - 1][state.y] != -1 and self.grid[state.x][state.y + 1] != -1:
            actions.append('RU')

        return actions

    # Checks whether a point is within the grid borders.
    def is_point_within_borders(self, x, y):
        if x >= 0 and x < len(self.grid[0]) and y >= 0 and y < len(self.grid):
            return True
        else:
            return False

    # Returns a state which consist of x, y coordinates by given action.
    def get_xy_by_state_action(self, state, action):
        if action == 'R':
            return (state.x, state.y + 1)
        elif action == 'RD':
            return (state.x + 1, state.y + 1)
        elif action == 'D':
            return (state.x + 1, state.y)
        elif action == 'LD':
            return (state.x + 1, state.y - 1)
        elif action == 'L':
            return (state.x, state.y - 1)
        elif action == 'LU':
            return (state.x - 1, state.y - 1)
        elif action == 'U':
            return (state.x - 1, state.y)
        elif action == 'RU':
            return (state.x - 1, state.y + 1)

    # Receives state and action, returns a new state by given action.
    def succ(self, state, action):
        x, y = self.get_xy_by_state_action(state, action)
        return Point(x, y)

    # Checks whether the current state is the goal.
    def is_goal(self, state):
        return state == self.goal

    # Receives state and action, returns the cost of moving from the current state to new state by given action.
    def step_cost(self, state, action):
        x, y = self.get_xy_by_state_action(state, action)
        return self.grid[x][y]

    # Takes the action and returns his level, as defined in the exercise, 'R' is first and 'RU' is the last, clock wise.
    @staticmethod
    def get_action_level(action):
        if action == 'R':
            return 1
        elif action == 'RD':
            return 2
        elif action == 'D':
            return 3
        elif action == 'LD':
            return 4
        elif action == 'L':
            return 5
        elif action == 'LU':
            return 6
        elif action == 'U':
            return 7
        elif action == 'RU':
            return 8



# This class implements a factory pattern in order to generate the algorithms.
# All the algorithm classes share the same interface to run their algorithm.
class AlgorithmFactory(object):
    # This function acts as a Factory Pattern which gets as a string the algorithm and returns the required algorithm.
    @staticmethod
    def get_algorithm(chosen_algorithm, problem):
        if chosen_algorithm == None or problem == None:
            raise Exception('Algorithm or Problem are null.')

        if chosen_algorithm == 'IDS':
            return IDS(problem)
        elif chosen_algorithm == 'UCS':
            return UniformCostSearch(problem)
        elif chosen_algorithm == 'ASTAR':
            return AStarSearch(problem)
        elif chosen_algorithm == 'IDASTAR':
            return IDAStarSearch(problem)
        else:
            raise Exception('Invalid algorithm type.')



# A node class which represents a current state that has been expanded by the algorithm and the problem.
# This class saves all the necessary items in order to correctly represent the current node.
@total_ordering
class Node(object):
    # Initialization the necessary items to describe the current state.
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        self.discovery_time = 0

    # A function that checks where I can expand from the current node, I call to functions from Problem class.
    def expand(self, problem):
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    # Takes the problem and the state (which is next one) and constructs a valid Node object.
    def child_node(self, problem, action):
        next_state = problem.succ(self.state, action)
        return Node(next_state, self, action, self.path_cost+problem.step_cost(self.state, action))

    # Takes the path and extracts what actions was made.
    def solution(self):
        return [node.action for node in self.path()[1:]]

    # Iterates from the current node to the start node and creates a list of nodes which lives in that path.
    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # There is a specific constricts how I define a lower than.
    # The current node cost is less or equal to the other node's cost, therefore current node will be before the other node.
    # The current node expended time is less or equal to the other node's expanded time, therefore the current node will be before the other node.
    # Current node movement (R, D, RU...) is lower than the other node's movnemt, I defined a priority function that handles this.
    def __lt__(self, node):
        return self.path_cost <= node.path_cost and self.discovery_time <= node.discovery_time and GridSearchProblem.get_action_level(self.action) < GridSearchProblem.get_action_level(node.action)

    # Equality implementation for this class.
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    # Not equality implementation for this class.
    def __ne__(self, other):
        return not (self == other)

    # Hash implementation for this class.
    def __hash__(self):
        return hash(self.state)



# This is Priority Queue implementation of a queue with some specifics I want.
# I pass a function that defines how the items inside will be prioritized, it behaves like a regular queue but with special order.
class PriorityQueue(object):

    # Defining the list for the queue implementation and a function which defines the property.
    def __init__(self, f=lambda x: x):
        self.heap = []
        self.f = f

    # Takes the list and applies the function on the item and inserts it to the queue.
    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), item))

    # Takes a list of items and adds them to the queue using the append function.
    def extend(self, items):
        for item in items:
            self.append(item)

    # Takes out the next element from the queue.
    # The element that will come out is the element that satisfy the function constraints.
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    # Returns the length of the queue.    
    def __len__(self):
        return len(self.heap)

    # Iterates on the elements in the queue and checks if there is an element that matches one of the items in the queue.
    def __contains__(self, key):
        return any([item == key for _, item in self.heap])

    # Iterates on the elements in the queue and returns the required element by the given key.
    def __getitem__(self, key):
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    # Removes an element from the queue by the given key.
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

    # Converts the queue to a list which represents as a string and returns it.
    def __repr__(self):
      return str(self.heap)



# All the algorithms share saving the problem and the pop counter.
# Therefore this abstract algorithm class will save these values.
class AlgorithmsAbstract(object):
    # The constructor receives some arguments and saves them.
    def __init__(self, problem):
        self.problem = problem
        self.pop_counter = 0

    # Define a function like an interface that every algorithm must implement.
    def run(self):
        pass



# IDS Algorithm implementation, there are 2 functions to notice her.
# "Run" function which is the algorithm of the depth, and the other function is the search algorithm itself.
class IDS(AlgorithmsAbstract):
    # Initialize the constructor of the inherited class. (Parent constructor initialization)
    def __init__(self, problem):
        super().__init__(problem)

    # Runs IDS Algorithm, calculates the maximum depth and then performs for loop and increases the depth each time.
    def run(self):
        for depth in range(0, 21):
            solution, path_cost, pop_counter = self.depth_limited_search(self.problem, depth)

            if solution:
                return solution, path_cost, pop_counter

        return None, None, None

    # Runs the algorithm of IDS until specific depth which is defined from the "run" function.
    def depth_limited_search(self, problem, depth):
        frontier = [(Node(problem.start))]

        while frontier:
            node = frontier.pop()

            if problem.is_goal(node.state):
                return node.solution(), node.path_cost, self.pop_counter

            self.pop_counter += 1

            if node.depth < depth:
                expanded_list = node.expand(problem)
                frontier.extend(expanded_list[::-1])

        return None, None, None



# This is an abstract class which inherits from algorithm class.
# This class purpose is to be an abstract class of all the weighted search algorithms.
# The search algorithms that inherits from this one are UCS, A* and IDA*.
class WeightedAlgorithmsAbstract(AlgorithmsAbstract):
    # Initialize the constructor of the inherited class. (Parent constructor initialization)
    def __init__(self, problem):
        super().__init__(problem)

    # This function is looking backward and returning the sum of weights until this node from the beginning.
    def g(self, node):
        return node.path_cost

    # This is implementation of Chebyshev Distance, check out the proof at the PDF to understand more.
    def h(self, node):
        current_point = node.state
        goal_point = self.problem.goal

        return max(abs(current_point.x - goal_point.x), abs(current_point.y - goal_point.y))

    # Combination of G function and H function.
    def g_h(self, n):
        return self.g(n) + self.h(n)

    # Algorithm implementation for BFGS.
    def best_first_graph_search(self, problem, f):
        node = Node(problem.start)
        frontier = PriorityQueue(f)
        frontier.append(node)
        closed_list = set()
        identifier_counter = 0

        while frontier:
            node = frontier.pop()

            if problem.is_goal(node.state):
                return node.solution(), node.path_cost, self.pop_counter

            closed_list.add(node.state)
            self.pop_counter += 1

            for child in node.expand(problem):
                identifier_counter += 1
                child.discovery_time = identifier_counter

                if child.state not in closed_list and child not in frontier:
                    frontier.append(child)
                elif child in frontier and f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)

        return None, None, None



# Uniform Cost Search Algorithm implementation.
# One function to notice, the "Run" algorithm which start BFGS from inherited class.
class UniformCostSearch(WeightedAlgorithmsAbstract):
    # Initialize the constructor of the inherited class. (Parent constructor initialization)
    def __init__(self, problem):
        super().__init__(problem)

    # Calls BFGS with "g" function which is defined in the inherited class.
    def run(self):
        return self.best_first_graph_search(self.problem, f=self.g)



# A* Algorithm implementation.
# One function to notice, the "Run" algorithm which start BFGS from inherited class.
class AStarSearch(WeightedAlgorithmsAbstract):
    # Initialize the constructor of the inherited class. (Parent constructor initialization)
    def __init__(self, problem):
        super().__init__(problem)

    # Calls BFGS with "g" function and "h" function which is defined in the inherited class.
    def run(self):
        return self.best_first_graph_search(self.problem, f=self.g_h)



# IDA* Algorithm implementation, there are 2 functions to notice her.
# "Run" function which is the algorithm of the depth, and the other function is the BFGS algorithm itself with depth limit.
class IDAStarSearch(WeightedAlgorithmsAbstract):
    # Initialize the constructor of the inherited class. (Parent constructor initialization)
    def __init__(self, problem):
        super().__init__(problem)
        self.new_limit = None
        self.current_depth = 0

    # This is function is the algorithm which starts a run of DFS_F at every new iteration.
    def run(self):
        start_node = Node(self.problem.start)
        end_node = Node(self.problem.goal)
        self.lower_limit_cost = self.h(start_node)

        while self.current_depth < 21:
            current_iteration_cost = self.lower_limit_cost
            self.lower_limit_cost = math.inf

            solution, path_cost, pop_counter = self.dfs_f(start_node, 0, current_iteration_cost, end_node)

            if solution:
                return solution, path_cost, pop_counter

        return None, None, None

    # This algorithm performs DFS_F run from the current node until the 'current_iteration_cost' and it attempts to reach to the goal.
    def dfs_f(self, current_node, path_cost_until_now, current_iteration_cost, goal):
        current_cost = path_cost_until_now + self.h(current_node)

        if current_cost > current_iteration_cost:
            self.lower_limit_cost = min(self.lower_limit_cost, current_cost)
            return None, None, None
        
        if self.problem.is_goal(current_node.state):
            return current_node.solution(), current_node.path_cost, self.pop_counter

        self.pop_counter += 1

        for child in current_node.expand(self.problem):
            child_cost = self.problem.grid[child.state.x][child.state.y]

            # The algorithm started to explore some new depth and therefore I increase the counter.
            # Then if I reached to 21 I will stop the algorithm.
            self.current_depth += 1

            if self.current_depth > 20:
                return None, None, None

            solution, path_cost, pop_counter = self.dfs_f(child, path_cost_until_now + child_cost, current_iteration_cost, goal)

            # Since the algorithm isn't going inside anymore for this path.
            # I want to descrase the counter by 1 in other to find a maxiumu depth of 20.
            self.current_depth -= 1


            if solution:
                return solution, path_cost, pop_counter

        return None, None, None



# Responsible to take the final outputs from the algorithm and to output it as the result of the run.
def solution_output(file_name, solution, path_cost, pop_counter):
    solution_output_string = ''

    if solution == None:
        solution_output_string = 'no path'
    else:
        solution = '-'.join(solution)
        solution_output_string = f'{solution} {path_cost} {pop_counter}'

    print(solution_output_string)
    with open(file_name, 'w') as f:
        f.write(solution_output_string)



# Loads the input.txt file, parse the arguments to object, gets the required algorithm, runs the algorithm, and sends the result to output.
def main():
    input_manager = InputManager('./input.txt')
    input_manager.load()

    problem = GridSearchProblem(input_manager)

    algorithm = AlgorithmFactory.get_algorithm(input_manager.algorithm, problem)
    solution, path_cost, pop_counter = algorithm.run()

    solution_output('./output.txt', solution, path_cost, pop_counter)



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)