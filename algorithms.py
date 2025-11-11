import collections
import itertools
import time
import random
import math
import functools
import json

from concurrent.futures.thread import ThreadPoolExecutor

DISABLED_PROBLEMS = [
    'backtracking',
    'backtracking+mrv',
    'simulated_annealing',
    'simulated_annealing+geometric_cooling+dt+dr0_995',
    'simulated_annealing+geometric_cooling+dt+dr0_7',
    'simulated_annealing+geometric_cooling+ft1_0+dr0_995',
    'simulated_annealing+geometric_cooling+ft1_0+dr0_7',
    'genetic_algorithm'
]

def all_problems():
    queen_sizes = [
        8, 10, 12, 14,
        16, 18, 20, 22,
        24, 26, 28, 30,
        32, 64, 65, 67,
        80, 90, 100, 128
    ]

    default_timeout = 60
    default_size = 8
    default_iterations = 5_000
    default_rate = 0.995

    # Calculate the scaling factor using square root.
    # Formula: timeout = default_timeout * pow(size, 1) / pow(default_size, 1).
    def timeout(size, base_size=default_size, base_timeout=default_timeout):
        if size <= base_size:
            return base_timeout
        
        scaling_factor = pow(size, 1) / pow(base_size, 1)
        return round(base_timeout * scaling_factor)
    
    # Calculate the scaling factor using the power of 2.
    # Formula: iterations = default_iterations * pow(size, 2) / pow(default_size, 2).
    def iterations(size):
        if size <= default_size:
            return default_iterations
        
        scaling_factor = pow(size, 2) / pow(default_size, 2)
        return round(default_iterations * scaling_factor)
    
    # Calculates a starting temperature based on problem size.
    # Larger problems need more energy to explore.
    def temperature(size):
        return max(1.0, size / 2.0)
    
    # Calculates a cooling rate based on problem size.
    # Larger problems need to cool more slowly.
    def cooling_rate(size, base_size=default_size, base_rate=default_rate):

        # Assumes that the gap from 1.0 is inversely
        # proportional to the size of the problem.
        # Formula: rate = 1.0 - (cooling_constant / size).
        def cooling_constant():
            epsilon_base = 1.0 - base_rate
            return epsilon_base * base_size

        # Use the base_rate for any size at or below the base size.
        if size <= base_size:
            return base_rate
        
        # Calculate the rate using the formula and make sure
        # the rate never exceeds 1.0.
        return min(1.0 - (cooling_constant() / size), 0.99999999)

    # All different approaches that will try to solve the N-Queen problem.
    problems = {
        'backtracking': [],
        'backtracking+mrv': [],
        'simulated_annealing': [],
        'simulated_annealing+geometric_cooling+dt+dr0_995': [],
        'simulated_annealing+geometric_cooling+dt+dr0_7': [],
        'simulated_annealing+geometric_cooling+ft1_0+dr0_995': [],
        'simulated_annealing+geometric_cooling+ft1_0+dr0_7': [],
        'genetic_algorithm': []
    }

    for size in queen_sizes:
        problems['backtracking'].append({
            'size': size,
            'timeout': timeout(size)
        })

        problems['backtracking+mrv'].append({
            'size': size,
            'timeout': timeout(size)
        })

        problems['simulated_annealing'].append({
            'size': size,
            'iterations': iterations(size),
            'timeout': timeout(size, base_timeout=10)
        })

        problems['simulated_annealing+geometric_cooling+dt+dr0_995'].append({
            'size': size,
            'iterations': float('inf'),
            'temperature': temperature(size),
            'rate': cooling_rate(size),
            'timeout': timeout(size, base_timeout=10)
        })

        problems['simulated_annealing+geometric_cooling+dt+dr0_7'].append({
            'size': size,
            'iterations': float('inf'),
            'temperature': temperature(size),
            'rate': cooling_rate(size, base_rate=0.7),
            'timeout': timeout(size, base_timeout=10)
        })

        problems['simulated_annealing+geometric_cooling+ft1_0+dr0_995'].append({
            'size': size,
            'iterations': float('inf'),
            'temperature': 1.0,
            'rate': cooling_rate(size),
            'timeout': timeout(size, base_timeout=10)
        })

        problems['simulated_annealing+geometric_cooling+ft1_0+dr0_7'].append({
            'size': size,
            'iterations': float('inf'),
            'temperature': 1.0,
            'rate': cooling_rate(size, base_rate=0.7),
            'timeout': timeout(size, base_timeout=10)
        })

        problems['genetic_algorithm'].append({
            'size': size,
            'k': round(pow(size, 0.80)),
            'iterations': float('inf'),
            'mutation_rate': 0.003,
            'timeout': timeout(size)
        })

    return problems


# The positions of the queen on the chessboard.
Pair = collections.namedtuple('Pair', ['i', 'j'])

def in_same_row(q1, q2):
    return q1.i == q2.i

def in_same_column(q1, q2):
    return q1.j == q2.j

# Efficient computation of the check to find out if
# two queens are on the same diagonal.
def in_same_diagonal(q1, q2):
    delta_i = abs(q1.i - q2.i)
    delta_j = abs(q1.j - q2.j)

    return delta_i == delta_j

# Returns True when the two given queens attack
# each other, False otherwise.
def attacking_each_other(q1, q2):
    conditions = [
        in_same_row(q1, q2),
        in_same_column(q1, q2),
        in_same_diagonal(q1, q2)
    ]

    return len(list(filter(lambda x: x, conditions))) != 0

# Generates an array of integers from a to b,
# returning the first element after a shuffle.
def random_int(a, b):
    values = [i for i in range(a, b + 1)]
    random.shuffle(values)
    return values[0]

# Executes the specified function with a timeout value.
def exec(f, timeout=None):
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(f)
            return future.result(timeout)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return None


# The general framework for CSP problems. Values are chosen for one
# variable at a time, and the process is reversed when a variable
# runs out of valid values to assign.
class CSP_BacktrackingSearchFramework:
    def __init__(self, csp_problem, reporter=None):
        self.csp_problem = csp_problem
        self.reporter = reporter

    def run(self):
        base_assignment = self.csp_problem.initialize_state()
        self.__backtrack(base_assignment)
        return base_assignment

    def __backtrack(self, assignment):
        # This means that the algorithm will stop only when a
        # complete solution is found, that is, when every variable
        # has been assigned and they are all consistent with each other.
        if self.csp_problem.is_complete(assignment):
            return assignment

        # Select a variable from the list, which is the state that keeps
        # track of assigned and unassigned variables.
        variable = self.csp_problem.select_unassigned_variable(assignment)

        # Count of the number of nodes visited, meaning that
        # a different configuration is being explored.
        if self.reporter is not None:
            self.reporter.counter('nodes_expanded', 1)
    
        # A value is chosen from the domain of the selected variable.
        for value in self.csp_problem.order_domain_values(variable, assignment):
            if self.csp_problem.is_consistent(variable, value, assignment):
                self.csp_problem.assign_variable(variable, value, assignment)
                inferences = self.csp_problem.inferences(variable, value, assignment)

                if self.csp_problem.is_inferences_valid(inferences):
                    self.csp_problem.assign_inferences(inferences, assignment)
                    result = self.__backtrack(assignment)

                    if result != self.csp_problem.failure():
                        return result

                # This is the number of wrong choices with inconsistent solutions.
                if self.reporter is not None:
                    self.reporter.counter('removes', 1)

                # When the backtracking algorithm returns an incomplete result,
                # the chosen path cannot lead to a valid solution.
                # Therefore all available values for the variable have been
                # tried, so the previous assignments are incorrect.
                # All inference is removed.
                self.csp_problem.remove_assigned_variable(variable, value, assignment)
                self.csp_problem.remove_assigned_inferences(inferences, assignment)

        # After clearing the state, an error is returned to inform that
        # previous choices must be changed.
        return self.csp_problem.failure()

class CSP_QueenProblem:
    def __init__(self, n_queens, reporter=None):
        self.n_queens = n_queens
        self.reporter = reporter

    def initialize_state(self):
        # The initial state is generated by assigning a fixed row
        # to each queen, leaving the column undefined.
        return {
            'assigned_queens': [],
            'unassigned_queens': [Pair(i=index, j=None) for index in range(0, self.n_queens)],
        }

    def is_complete(self, assignment):
        assigned = assignment['assigned_queens']
        unassigned = assignment['unassigned_queens']

        # The state is complete when all queens have been assigned.
        # When a new queen is added to the assigned list,
        # it is consistent with all the queens already present.
        return len(assigned) == self.n_queens and len(unassigned) == 0

    def select_unassigned_variable(self, assigment):
        # The first variable is selected from the list,
        # in no particular order.
        return assigment['unassigned_queens'][0]

    def order_domain_values(self, variable, assigment):
        # All available values are returned, even
        # potentially inconsistent ones, in no particular order.
        for j in range(0, self.n_queens):
            yield Pair(variable.i, j)

    def is_consistent(self, variable, value, assignment):
        # The queen (i, j) that was chosen for the given variable (Q_ij)
        q1 = value

        for q2 in assignment['assigned_queens']:
            # Count how many comparisons the CSP algorithm is performing.
            # This is an approximation: attacking_each_other is constant,
            # so instead of counting all three different logical operations,
            # the function call is considered
            if self.reporter is not None:
                self.reporter.counter('comparisons', 1)
            
            # The cost of the check is constant.
            if attacking_each_other(q1, q2):
                return False

        return True

    def assign_variable(self, variable, value, assignment):
        # The full queen (value) is added to the list of assigned
        # queens, while the variable is removed (unassigned queen).
        assignment['assigned_queens'].append(value)
        assignment['unassigned_queens'].remove(variable)

    def inferences(self, variable, value, assignment):
        pass

    def is_inferences_valid(self, inferences):
        return True

    def assign_inferences(self, inferences, assignment):
        pass

    def remove_assigned_variable(self, variable, value, assignment):
        # The full queen is removed and the variable is added
        # to the beginning of the list, to respect the original order.
        if value in assignment['assigned_queens']:
            assignment['assigned_queens'].remove(value)

        assignment['unassigned_queens'].insert(0, variable)

    def remove_assigned_inferences(self, inferences, assignment):
        pass

    def failure(self):
        # A state without queens.
        return {
            'assigned_queens': [],
            'unassigned_queens': []
        }

class CSP_QueenProblemMRV(CSP_QueenProblem):
    def __init__(self, n_queens, reporter=None):
        super().__init__(n_queens, reporter)

    def select_unassigned_variable(self, assigment):
        assigned = assigment['assigned_queens']
        unassigned = assigment['unassigned_queens']

        def count_legal_moves(variable):
            # Counter to keep track of the total number of moves.
            total = 0

            # Iterate through all available columns to measure
            # how many legal moves the given variable has.
            for column in range(self.n_queens):
                q1 = Pair(variable.i, column)
                is_legal = True

                # After assigning a column, check to see if there is
                # at least one attacking queen. If there is an
                # attacking queen, it cannot be considered a legal move.
                for q2 in assigned:
                    # Further comparisons to select the most appropriate order.
                    if self.reporter is not None:
                        self.reporter.counter('comparisons', 1)

                    if attacking_each_other(q1, q2):
                        is_legal = False
                        break
                
                # Increase the counter only if the selected column
                # is consistent with all other queens already assigned.
                if is_legal:
                    total = total + 1

            return total

        # Keep track of the remaining minimum values.
        min_moves = float('inf')
        current_min = None

        for variable in unassigned:
            current_moves = count_legal_moves(variable)

            if current_moves < min_moves:
                min_moves = current_moves
                current_min = variable

        return current_min

class SimulatedAnnealingFramework:
    def __init__(self, problem, reporter=None):
        self.problem = problem
        self.reporter = reporter

    def run(self):
        current_state = self.problem.initial_state()

        for iteration in itertools.count(1):
            # The scheduling function determines how quickly
            # the algorithm will reject the solution with the
            # lowest energy.
            t = self.problem.schedule(iteration)

            if t == 0 or self.problem.is_enough(current_state):
                if self.reporter is not None:
                    self.reporter.counter('iterations', iteration)

                return [current_state, self.problem.energy(current_state)]

            # For simulated annealing, successors must be complete,
            # meaning all variables involved in the problem must
            # have an associated value.
            successors = self.problem.successors(current_state)

            # The number of successors generated for each node.
            if self.reporter is not None:
                self.reporter.counter('successors', len(successors))
                
            random.shuffle(successors)
            next_state = successors[0]

            current_energy = self.problem.energy(current_state)
            next_energy = self.problem.energy(next_state)

            difference = next_energy - current_energy

            if difference < 0:
                current_state = next_state
            else:
                exponent = -difference / t
                if random.random() < math.exp(exponent):
                    current_state = next_state

class SimulatedAnnealingQueenProblem:
    def __init__(self, n_queens, iterations, reporter=None):
        self.iterations = iterations
        self.n_queens = n_queens
        self.reporter = reporter

    def initial_state(self):
        # The initial state is a random complete state.
        N = self.n_queens
        return [Pair(i, random_int(0, N - 1)) for i in range(0, N)]

    def schedule(self, iteration):
        return math.log(self.iterations / iteration)

    def is_enough(self, current_state):
        return self.energy(current_state) == 0

    def successors(self, state):
        N = self.n_queens
        next_states = []

        # Each queen is moved along the columns,
        # resulting in N * (N - 1) successes for each node.
        for queen in state:
            without_queen = list(filter(lambda x: x != queen, state))

            node = Pair(queen.i, (queen.j + 1) % N)
            while node != queen:
                next_states.append(sorted(without_queen + [node], key=lambda x: x.i))
                node = Pair(queen.i, (node.j + 1) % N)

        return next_states;

    def energy(self, state):
        # The overall energy is determined by the number
        # of pairs attaching to each other.
        attacking_positions = set()

        for q1 in state:
            aggressive_queens = list(filter(lambda x: q1 != x, state))

            for q2 in aggressive_queens:
                if attacking_each_other(q1, q2):
                    attacking_positions.add(tuple(sorted([q1.i, q2.i, q1.j, q2.j])))

        return len(attacking_positions)

class SimulatedAnnealingQueenProblemGeometricCooling(SimulatedAnnealingQueenProblem):
    def __init__(self, n_queens, iterations, temperature, rate, reporter=None):
        super().__init__(n_queens, iterations, reporter)
        self.temperature = temperature
        self.rate = rate

    # Based on geometric cooling technique.
    def schedule(self, iteration):
        if iteration > self.iterations:
            return 0

        self.temperature = self.temperature * self.rate
        return self.temperature

class GeneticAlgorithmFramework:
    def __init__(self, problem, reporter=None):
        self.problem = problem
        self.reporter = reporter

    def run(self):
        population = self.problem.initial_population()

        for iteration in itertools.count(1):
            # At each iteration the new generation
            # is used to prepare matching pairs.
            new_population = []

            # A matching dictionary is used to identify previously used
            # crossover points, thus avoiding having the same
            # configuration more than once in the new population.
            matches = {}

            # The number of iterations is equal to the
            # cardinality of the population.
            for _ in population:
                # The most valuable queen are selected according to a
                # specific fitness function.
                x = self.problem.pop_by_fit(population)
                y = self.problem.pop_by_fit(list(filter(lambda node: node != x, population)))

                # The selected match.
                match = tuple(sorted([x, y]))

                if matches.get(match) is None:
                    matches[match] = []

                # All individuals already generated by the
                # parent are retrieved for this run.
                siblings = matches[match]

                # A new child and the selected crossover point
                # are produced by calling the reproductive function.
                child, crossoverpoint = self.__reproduce(
                    first=x,
                    second=y,
                    individuals=siblings
                )

                # The new crossover point is recorded to reduce
                # the likelihood of choosing the same crossover point again.
                siblings.append(crossoverpoint)

                # The mutation is applied based on a specific probability distribution (e.g. Poisson).
                if self.problem.should_mutate(child, siblings=siblings):
                    if self.reporter is not None:
                        self.reporter.counter('mutations', 1)
                        
                    child = self.problem.apply_mutation(child, siblings=siblings)

                new_population.append(child)

            population = new_population

            # The algorithm will stop as soon as the solution
            # is sufficient or too much time has passed.
            if self.problem.is_enough(population, iteration):
                best = self.problem.best_of(population)

                if self.reporter is not None:
                    reporter = self.reporter
                    reporter.counter('iterations', iteration)
                    reporter.set_result('best_solution', best[0])
                    reporter.set_result('best_fitness', best[1])

                return best


    def __reproduce(self, first, second, individuals=[]):
        n = len(first)

        # A crossover point is chosen also taking into
        # account the given individuals.
        crossover_points = list(filter(lambda x: x not in individuals, [i for i in range(1, n)]))
        random.shuffle(crossover_points)

        # If all possible crossover points have been generated, a random point is chosen.
        point = random_int(1, n - 1) if len(crossover_points) == 0 else crossover_points[0]
        child = self.problem.cross(first, second, point)
        
        return [child, point]

class GeneticAlgorithmQueenProblem:
    def __init__(self, n, k, iterations, mutation_rate, reporter=None):
        self.n = n
        self.k = k
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.reporter = reporter

    def initial_population(self):
        # The genetic algorithm works with strings.
        # A set of chessboards with assigned queens is randomly generated.
        states = [0 for _ in range(0, self.k)]
        states = list(map(lambda _: [str(random_int(0, self.n - 1)) for i in range(0, self.n)], states))
        states = list(map(lambda l: '_'.join(l), states))

        # This is an array of strings, which is a string
        # representation of the chessboard with assigned queens.
        return states

    def pop_by_fit(self, population):
        with_fitness = self.__queens_with_fitness(population)

        # Calculating the probability of being chosen for
        # the reproductive process.
        fitsum = functools.reduce(lambda x, y: x + y.j, with_fitness, 0)
        with_fitness = list(map(lambda p: Pair(p.i, p.j / fitsum), with_fitness))

        # The most valuable string is chosen based on
        # the fitness function within the given population.
        choices = list(map(lambda p: p.i, with_fitness))
        probabilities = list(map(lambda p: p.j, with_fitness))

        # The most valuable state is selected based on the
        # probabilities involved. The string representation
        # is then returned.
        bet = random.choices(choices, weights=probabilities, k=1)[0]
        return '_'.join(map(lambda p: str(p.j), bet))

    def should_mutate(self, child, siblings):
        def poisson(l, n):
            return math.exp(-l) * ((l**n) / math.factorial(n))

        mutation_rate = self.mutation_rate * (len(siblings) + 1) * (siblings.count(child) + 1)
        return random.random() < poisson(l=mutation_rate, n=1)

    # The state is changed by changing the position
    # of a queen with respect to the available columns.
    def apply_mutation(self, child, siblings):
        # A random position is chosen.
        position = random_int(0, self.n - 1)

        # The string representation is mapped to the chess board
        # representation.
        queen_state = self.__str_to_queen_state(child)

        # Selection of the queen which will change its position
        # in the column.
        queen_to_change = queen_state[position]
        old_column = queen_to_change.j

        # Randomly choose a new value for the column.
        values = list(filter(lambda x: x != old_column, [i for i in range(self.n)]))
        random.shuffle(values)
        value = values[0]

        # Modify the queen accordingly and return the
        # new string representation.
        new_queen = Pair(position, value)
        queen_state[position] = new_queen

        return self.__queen_state_to_str(queen_state)

    def is_enough(self, population, iteration):
        def arithmetic_series(start,  step,  end):
            sum = (end / 2) * (2 * start + (end - 1) * step)
            return sum

        # Whenever the current iteration exceeds the maximum
        # number of iterations, the algorithm should stop.
        if iteration > self.iterations:
            return True

        # Whenever the population is composed of the
        # same element, the algorithm should stop.
        if population.count(population[0]) == len(population):
            return True

        # Whenever there is at least one state with maximum
        # fit, the algorithm should stop.
        max_fit = round(arithmetic_series(1, 1, self.n - 1))
        if self.reporter is not None:
            self.reporter.set_result('fit_to_reach', max_fit)

        with_fitness = self.__queens_with_fitness(population)
        return len(list(filter(lambda x: x.j >= max_fit, with_fitness))) != 0

    def best_of(self, population):
        with_fitness = self.__queens_with_fitness(population)
        pair = max(with_fitness, key=lambda x: x.j)
        return [self.__queen_state_to_str(pair.i), pair.j]

    def cross(self, first, second, point):
        first_queen_board = self.__str_to_queen_state(first)
        second_queen_board = self.__str_to_queen_state(second)

        new_board = []
        for index in range(self.n):
            if index < point:
                new_board.append(first_queen_board[index])
            else:
                new_board.append(second_queen_board[index])

        return self.__queen_state_to_str(new_board)

    # Mapping from chessboard representation to string representation.
    def __queen_state_to_str(self, queen_state):
        return '_'.join(map(lambda pair: str(pair.j), queen_state))

    # From the population (array of strings),
    # to an array of tables with queens represented as pairs.
    # Each entry also has its own fitness value.
    def __queens_with_fitness(self, population):
        def population_to_queens():
            queens = list(map(lambda s: s.split('_'), population))
            queens = list(map(lambda l: list(map(lambda x: Pair(x[0], int(x[1])), enumerate(l))), queens))
    
            return queens

        # The fitness function is the number of non-attacking queens.
        def fitness(state):
            non_attacking_positions = set()
    
            for q1 in state:
                non_aggressive_queens = list(filter(lambda x: q1 != x, state))
    
                for q2 in non_aggressive_queens:
                    if not attacking_each_other(q1, q2):
                        non_attacking_positions.add(tuple(sorted([q1.i, q2.i, q1.j, q2.j])))
    
            return len(non_attacking_positions)

        with_fitness = []
        queens = population_to_queens()

        for node in queens:
            with_fitness.append(Pair(node, fitness(node)))

        return with_fitness

    # Mapping from string representation to chessboard representation.
    def __str_to_queen_state(self, string):
        columns = string.split('_')
        queen_state = list(map(lambda x: Pair(x[0], int(x[1])), enumerate(columns)))

        return queen_state

class Report:
    def __init__(self, name, problem_size=None, params=None):
        self.name = name
        self.problem_size = problem_size
        self.params = params or {}
        self.counters = collections.Counter()
        self.timing = {}
        self.results = {}

    def measure(self, key=None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                k = key or func.__name__
                self.timing.setdefault(k, []).append(elapsed)
                return result
            return wrapper
        return decorator

    def counter(self, name, amount=1):
        self.counters[name] += amount

    def set_result(self, key, value):
        self.results[key] = value

    def summary(self):
        return {
            'name': self.name,
            'problem_size': self.problem_size,
            'params': self.params,
            'timings': {k: {'calls': len(v), 'total': sum(v), 'avg': sum(v)/len(v)} for k,v in self.timing.items()},
            'counters': dict(self.counters),
            'results': self.results
        }

    def pretty_print(self):
        s = self.summary()

        print(f"=== Report: {s['name']} ===\n")
        print(f"Problem size: {s['problem_size']}")
        print(f"Params: {s['params']}\n")
        print("Timings:")

        for k,v in s['timings'].items():
            print(f"  {k}: calls={v['calls']}, total={v['total']:.6f}s, avg={v['avg']:.6f}s")

        print("\nCounters:")
        for k,v in s['counters'].items():
            print(f"  {k}: {v}")

        print("\nResults:")
        for k,v in s['results'].items():
            print(f"  {k}: {v}")

        print("="*50)

def benchmark():
    def append_to_jsonl(filepath, data):
        with open(filepath, 'a', encoding='utf-8') as f:
            # Serialize the dictionary to a JSON string and
            # write the JSON string followed by a newline character.
            json_line = json.dumps(data)
            f.write(json_line + '\n')

    def run(framework, reporter, timeout):
        result = exec(reporter.measure('run')(framework.run), timeout=timeout)
        
        if result is None:
            reporter.set_result('failure', 'Timeout occurred: > ' + str(timeout) + 's')
        else:
            reporter.set_result('solution', result)

        # ...
        append_to_jsonl(
            filepath=reporter.name + '.jsonl',
            data=reporter.summary()
        )

    for key, values in all_problems().items():

        if DISABLED_PROBLEMS.count(key):
            continue

        for params in values: 

            size = params['size']
            timeout = params['timeout']

            reporter = Report(key, problem_size=size, params=params)

            if ['backtracking', 'backtracking+mrv'].count(key) != 0:

                problem = None

                if key == 'backtracking':
                    problem = CSP_QueenProblem(size, reporter)
                else:
                    problem = CSP_QueenProblemMRV(size, reporter)

                run(CSP_BacktrackingSearchFramework(problem, reporter), reporter, timeout)

            if ['simulated_annealing', 'simulated_annealing+geometric_cooling+dt+dr0_995', 'simulated_annealing+geometric_cooling+dt+dr0_7', 'simulated_annealing+geometric_cooling+ft1_0+dr0_995', 'simulated_annealing+geometric_cooling+ft1_0+dr0_7'].count(key) != 0:

                problem = None

                if key == 'simulated_annealing':
                    problem = SimulatedAnnealingQueenProblem(
                        n_queens=size,
                        iterations=params['iterations'],
                        reporter=reporter
                    )
                else:
                    problem = SimulatedAnnealingQueenProblemGeometricCooling(
                        n_queens=size,
                        iterations=params['iterations'],
                        temperature=params['temperature'],
                        rate=params['rate'],
                        reporter=reporter)
                    
                run(SimulatedAnnealingFramework(problem, reporter), reporter, timeout)

            if ['genetic_algorithm'].count(key) != 0:

                problem = GeneticAlgorithmQueenProblem(
                    size,
                    params['k'],
                    params['iterations'],
                    params['mutation_rate'],
                    reporter
                )

                run(GeneticAlgorithmFramework(problem, reporter), reporter, timeout)

benchmark()
