######################################
# Copyright(C) 2021 - 2023 COLIBRITD - ICB
#
# Developers :
#  - Hamza JAFFALI < hamza.jaffali@colibritd.com >
#  - Frédéric HOLWECK < frederic.holweck@utbm.fr >
#
# Version : 1.0
#
# This file is part of collaboration between COLIBRTD AND ICB.
#
# This production can not be copied and / or distributed without the express
# permission of COLIBRITD and ICB
#
######################################
import random

import numpy as np
import math
from scipy.optimize import minimize

# Number of players, number of answers, number of qubits
n = 3
# Number of possible basis states
two_power_n = 2 ** n


def generate_all_functions_Bn_to_B():
    """Generate all the binary functions from B_n = {0,1}^n to B = {0,1}
    Returns:
        an array containing all possible functions in binary representation
    """
    tab = []
    for i in range(2 ** two_power_n):
        tab.append(integer_to_binary_tab(i, two_power_n))
    return tab


def generate_all_functions_Bn_to_Bn():
    """Generate all the binary functions from B_n = {0,1}^n to B_n
    Returns:
        an array containing all possible functions in binary representation
    """
    tab = []
    for i in range(2 ** (2 * n)):
        tab.append(np.array_split(integer_to_binary_tab(i, 2 * n), n))
    return tab


def evaluate_function(f, f_variables):
    """ Evaluate a binary function on a specific set of binary variables
    Arguments:
        f: binary representation of the function to evaluate
        f_variables: vector of binary variables
    """
    # print(f_variables)
    # print(int(binary_tab_to_integer(f_variables)))
    return f[binary_tab_to_integer(f_variables)]


def evaluate_function_strategy(h, h_variables):
    """
    Arguments:
        h: binary representation of the strategy function to evaluate
        h_variables: vector of binary variables
    """

    return [h[k][h_variables[k]] for k in range(len(h_variables))]


def evaluate_problem(f, g, f_variables, g_variables):
    """ Evaluate the problem equation with the inputs
    Arguments:
        f: binary function representing the left-hand side of the problem's equation
        g: binary function representing the right-hand side of the problem's equation
        f_variables: vector of binary variables to evaluate f
        g_variables: vector of binary variables to evaluate g
    Returns:
        True if the equation given by f and g is satisfied, False otherwise
    """
    if evaluate_function(f, f_variables) == evaluate_function(g, g_variables):
        return 1
    return 0


def compute_score_classical_strategy(h, f, g):
    """ Computes the score of the classical strategy h for the problem given by (f,g)
    Arguments:
        h: strategy
        f: binary function representing the left-hand side of the problem's equation
        g: binary function representing the right-hand side of the problem's equation
    """
    somme = 0
    for i in range(two_power_n):
        param = integer_to_binary_tab(i, n)
        somme = somme + evaluate_problem(f, g, param, evaluate_function_strategy(h, param))
    score = somme / (two_power_n * 1.0)
    return score


def find_best_classical_strategies(f: list, g: list):
    """ Find the best classical strategy for the CHSH game defined by the inputs
    Arguments:
        f: defines the boolean equation with the referee questions
        g: defines the boolean equation with players answers
    Returns:
        best_score: the highest probability of winning the for the best classical strategy
        best_strategies: the strategies (deterministic description of the answers) that reach the best score
    """
    best_score = 0
    best_strategies = []
    all_classical_strategies = generate_all_functions_Bn_to_Bn()
    for index in range(len(all_classical_strategies)):
        test_score = compute_score_classical_strategy(all_classical_strategies[index], f, g)
        if test_score == best_score:
            best_strategies.append(all_classical_strategies[index])
        elif test_score > best_score:
            best_score = test_score
            best_strategies = [all_classical_strategies[index]]
    return best_score, best_strategies


### QUANTUM STRATEGIES ###


def construct_unitary_matrix(angles):
    """ Return a unitary matrix from the angles in parameter
        Arguments:
            angles : array of angles describing the angles of the unitary rotation
        """
    theta = angles[0]
    phi = angles[1]
    lambdaa = angles[2]
    return [[math.cos(theta / 2),                          -np.exp(lambdaa * 1j) * math.sin(theta / 2)],
            [np.exp(phi * 1j) * math.sin(theta / 2),        np.exp((phi + lambdaa) * 1j) * math.cos(theta / 2)]]


def apply_player_strategy(player_index, angles, state):
    """ Applies a rotation to the qubit owned by the player_index-th player.
    This rotation of the state models the local rotation of the local measurement basis of the corresponding player
    Args:
        player_index: integer representing the index of the player in the game
        angles: array of angles describing the player's new local basis where to measure this part of the state
        state: vector representing the quantum state shared by the players
    Returns:
        succeeded: a boolean indicating whether the function was successfully applied or not
    """
    state_copy = state.copy()

    identity_matrix = np.identity(2)
    rotation_matrix = construct_unitary_matrix(angles)

    global_matrix = 1
    for _ in range(0, player_index):
        global_matrix = np.kron(global_matrix, identity_matrix)
    global_matrix = np.kron(global_matrix, rotation_matrix)
    for _ in range(player_index + 1, n):
        global_matrix = np.kron(global_matrix, identity_matrix)

    return global_matrix.dot(state_copy)


def get_probabilities(state):
    """ Retrieve the probabilities associated with the input state
    Args:
        state: array representing the vector in the Hilbert space
    Returns:
        tab: an array containing all the probabilities associated
              with each amplitude of the state
    """
    return [abs(state[i])**2 for i in range(len(state))]


def evaluate_quantum_strategy_for_question(angles_array, state, f, g, f_variables):
    """ Evaluate the gain probability for the given set of angles for each player
        with respect to the question answered by the referee
    Arguments:
        angles_array: a 3D-array containing the the rotations angles,
                       for each question of the referee,
                       for each player of the game
        state: vector representing the quantum state shared by the players
        f: defines the boolean equation with the referee questions
        g: defines the boolean equation with players answers
        f_variables: the question given by the referee
    """
    state_copy = state.copy()
    # for each player, apply his quantum strategy
    for i in range(n):
        state_copy = apply_player_strategy(i, angles_array[i][f_variables[i]], state_copy)

    probas = get_probabilities(state_copy)

    somme = 0
    for i in range(two_power_n):
        param = integer_to_binary_tab(i, n)
        somme = somme + probas[i] * evaluate_problem(f, g, f_variables, param)

    return somme


def evaluate_quantum_strategy(angles_array, state, f, g):
    """ Evaluate the gain probability for the given set of angles for each player
        by computing the average gain for each question
    Args:
        angles_array: array containing the angles representing the strategy
        state: vector representing the quantum state shared by the players
        f: defines the boolean equation with the referee questions
        g: defines the boolean equation with players answers
    """
    somme = 0
    for i in range(two_power_n):
        param = integer_to_binary_tab(i, n)
        somme = somme + evaluate_quantum_strategy_for_question(angles_array, state, f, g, param)
    score = somme / (two_power_n * 1.0)
    return score


def find_best_quantum_strategy(f, g, state):
    """ Use a classical optimizer to find the best angles defining the quantum strategy
    Arguments:
        f: defines the boolean equation with the referee questions
        g: defines the boolean equation with players answers
        state: vector representing the quantum state shared by the players
    """
    # for n players, 2 basis, and 3 angles
    init_angles = np.array([np.random.rand() * 2 * math.pi for _ in range(n * 2 * 3)])

    def function_to_optimize(angles_flat):
        angles_not_flat = np.reshape(angles_flat, (n, 2, 3))
        return 1 - evaluate_quantum_strategy(angles_not_flat, state, f, g)

    def callback(param_list):
        current_E = 1 - function_to_optimize(param_list)
        print("current gain:", current_E)
        return None

    method = "BFGS"
    options_2 = {'disp': False, 'maxiter': 150, 'gtol': 1e-5}
    #options_2 = {'disp': False, 'maxiter': 400}
    opt = minimize(function_to_optimize, init_angles, method=method, options=options_2, callback=callback)
    return opt


def interesting_problem(f: list, g: list) -> bool:
    """ Determine if the problem define by f and g is interesting for us
    It should verify that :
        1. All players should have a question and an answer involved in both sides of the equation
        2. The left and right side of the equation are different
    Arguments:
        f: defines the boolean equation with the referee questions
        g: defines the boolean equation with players answers
    """
    if f == g:
        return False

    nb_one_f = f.count(1)
    nb_one_g = g.count(1)
    if n == 2:
        if nb_one_f * nb_one_g == 0 or nb_one_f == two_power_n or nb_one_g == two_power_n \
                or nb_one_f == n or nb_one_g == n:
            return False
        else:
            return True
    if n == 3:
        non_useful_cases = [0, 3, 5, 10, 12, 15, 17, 34, 48, 51, 60, 63, 68, 80, 85, 90, 95, 102, 119, 136, 153, 160,
                            165, 170, 175, 187, 192, 195, 204, 207, 221, 238, 240, 243, 245, 250, 252, 255]
        binary_table_non_useful_cases = [integer_to_binary_tab(non_useful_cases[i], two_power_n)
                                         for i in range(len(non_useful_cases))]
        if (f in binary_table_non_useful_cases) or (g in binary_table_non_useful_cases):
            return False
        else:
            return True


def integer_to_binary_tab(integer: int, size: int) -> list[int]:
    binary = format(integer, 'b')
    tab = [0] * size
    length = len(binary)
    for i in range(length):
        tab[size - length + i] = int(binary[i])
    return tab


def binary_tab_to_integer(binary_tab):
    size = len(binary_tab)
    return int(sum(binary_tab[size - i - 1] * 2 ** i for i in range(size)))


def normalize(v):
    """ 
    Normalize the vector given in parameter (norm 2)
    Args:
        v: vector, represented by a list or an array
    """
    norm = np.linalg.norm(v, ord=2)
    if norm == 0:
        return v
    else:
        return v / norm


### TESTS  ###
# Quantum system shared among the players
## For 2-qubits, it is the Bell state
bell_state = [0] * 4
bell_state[0] = 1
bell_state[-1] = 1
bell_state = normalize(bell_state)

## For 3-qubits,
### W
w_state = [0] * 8
w_state[1] = 1
w_state[2] = 1
w_state[4] = 1
w_state = normalize(w_state)

## For n-qubits,
### GHZ-GENERALIZED
ghz_n_state = [0] * two_power_n
ghz_n_state[0] = 1
ghz_n_state[-1] = 1
ghz_n_state = normalize(ghz_n_state)


# all_f = generate_all_functions_Bn_to_B()
# all_g = generate_all_functions_Bn_to_B()
#
#
# for i in range(len(all_f)):
#     for j in range(len(all_g)):
#         if interesting_problem(all_f[i], all_g[j]):
#             score_classical, best = find_best_classical_strategies(all_f[i], all_g[j])
#
#             if (score_classical < 0.77) and (score_classical > 0.73):
#                 best_quantum_w = find_best_quantum_strategy(all_f[i], all_g[j], w_state)
#                 score_quantum_w = 1 - best_quantum_w.fun
#                 best_quantum_ghz = find_best_quantum_strategy(all_f[i], all_g[j], ghz_n_state)
#                 score_quantum_ghz = 1 - best_quantum_ghz.fun
#
#                 if (score_quantum_w > score_classical) or (score_quantum_ghz > score_classical):
#                     print("f = ", all_f[i])
#                     print("g = ", all_g[j])
#                     print("score classique = ", score_classical)
#                     print("score quantique GHZ = ", score_quantum_ghz)
#                     print("score quantique W = ", score_quantum_w)
#                     print("\n")
#                     file_progression = open("results_ghz_compare_w.txt", "a")
#                     file_progression.write( "f = " + str(all_f[i]) +
#                                             "\ng = " + str(all_g[j]) +
#                                             "\nscore classique = " +
#                                             str(score_classical) +
#                                             "\nscore quantique GHZ = " +
#                                             str(score_quantum_ghz) +
#                                             "\nscore quantique W = " +
#                                             str(score_quantum_w) +
#                                             "\n\n")
#                     file_progression.close()
#

#### TEST For 2-qubit CHSH game

# f_test = [0, 0, 0, 1]
# g_test = [0, 1, 1, 0]
#
# best_quantum_bell = find_best_quantum_strategy(f_test, g_test, bell_state)
# score_quantum_bell = 1 - best_quantum_bell.fun
# print("Best quantum Bell :", score_quantum_bell)
#
# angles = np.reshape(best_quantum_bell.x, (n, 2, 3))
# print(angles)
#
#
# print(res)

#### TEST FOR 3-qubit CHSH GAME

# print("Game 1")
# f_test = [0, 1, 1, 1, 0, 0, 0, 1]
# g_test = [0, 1, 1, 0, 1, 0, 0, 1]
#
# score, strategies = find_best_classical_strategies(f_test, g_test)
# print("Best classical :", score)
# print("Best classical strategy :", strategies[0])
#
#
# best_quantum_ghz = find_best_quantum_strategy(f_test, g_test, ghz_n_state)
# score_quantum_ghz = 1 - best_quantum_ghz.fun
# print("Best quantum GHZ :", score_quantum_ghz)
#
# angles = np.reshape(best_quantum_ghz.x, (n, 2, 3))
# print(angles)
#
# #####################

# print("Game 2")
# f_test = [1, 0, 0, 0, 0, 0, 0, 1]
# g_test = [0, 1, 1, 0, 1, 0, 0, 1]
#
# # score, strategies = find_best_classical_strategies(f_test, g_test)
# # print("Best classical :", score)
# # print("Best classical strategy :", strategies[0])
#
# best_quantum_ghz = find_best_quantum_strategy(f_test, g_test, ghz_n_state)
# score_quantum_ghz = 1 - best_quantum_ghz.fun
# print("Best quantum GHZ :", score_quantum_ghz)

# best_quantum_w = find_best_quantum_strategy(f_test, g_test, w_state)
# score_quantum_w = 1 - best_quantum_w.fun
# print("Best quantum W :", score_quantum_w)
#
# angles = np.reshape(best_quantum_w.x, (n, 2, 3))
# print("W angles", angles)

# angles = np.reshape(best_quantum_ghz.x, (n, 2, 3))
# print("GHZ angles", angles)
#
# pi = np.pi
# a = pi/12
# b = pi/12+pi/2
# angles_debug = [[[pi/2, 0, a], [pi/2, 0, b]],
#                [[pi/2, 0, a], [pi/2, 0, b]],
#                [[pi/2, 0, a], [pi/2, 0, b]]]
#
# print(evaluate_quantum_strategy(angles_debug, ghz_n_state, f_test, g_test))
# print("000 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [0, 0, 0]))
# print("001 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [0, 0, 1]))
# print("010 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [0, 1, 0]))
# print("011 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [0, 1, 1]))
# print("100 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [1, 0, 0]))
# print("101 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [1, 0, 1]))
# print("110 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [1, 1, 0]))
# print("111 : ", evaluate_quantum_strategy_for_question(angles_debug, ghz_n_state, f_test, g_test, [1, 1, 1]))

# def generic_form_optimization(tab):
    # f_test = [1, 0, 0, 0, 0, 0, 0, 1]
    # g_test = [0, 1, 1, 0, 1, 0, 0, 1]
    # pi = math.pi
    #
    # theta_A_0 = pi/2
    # theta_B_0 = tab[1]
    # theta_C_0 = tab[2]
    # theta_A_1 = tab[3]
    # theta_B_1 = tab[4]
    # theta_C_1 = tab[5]
    #
    # phi_A_0 = tab[6]
    # phi_B_0 = tab[7]
    # phi_C_0 = tab[8]
    # phi_A_1 = tab[9]
    # phi_B_1 = tab[10]
    # phi_C_1 = tab[11]

    # lambda_A_0 = tab[0]
    # lambda_B_0 = tab[0]+pi/2
    # lambda_C_0 = tab[14]
    # lambda_A_1 = tab[15]
    # lambda_B_1 = tab[16]
    # lambda_C_1 = tab[17]
    # print(tab[0])

    # test_strategy = [[[theta_A_0, 0, lambda_A_0], [theta_A_0, 0, lambda_B_0]],
    #                  [[theta_A_0, 0, lambda_A_0], [theta_A_0, 0, lambda_B_0]],
    #                  [[theta_A_0, 0, lambda_A_0], [theta_A_0, 0, lambda_B_0]]]
    # return 1 - evaluate_quantum_strategy(test_strategy, ghz_n_state, f_test, g_test)

# def generic_form_evaluation(tab):
#     return 1 - generic_form_optimization(tab)
#
# init = [0 for _ in range(1)]
#
# method = "BFGS"
# options_2 = {'disp': False, 'maxiter': 1000}
# opt = minimize(generic_form_optimization, init, method=method, options=options_2, callback=None)
# print("TEST", opt.x, 1 - opt.fun)


#### TEST FOR 4-qubit CHSH GAME
# chsh_f_generic = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# chsh_g = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
# for i in range(1, 15):
#     for j in range(i+1, 16):
#             copy_f = chsh_f_generic.copy()
#             copy_f[i] = 1
#             copy_f[j] = 1
#
#             best_quantum = find_best_quantum_strategy(copy_f, chsh_g, ghz_n_state)
#             score_quantum = 1 - best_quantum.fun
#             if 0.85 < score_quantum < 0.86:
#                 print("f : ",copy_f)
#                 print("Best quantum :", score_quantum)
#                 score = find_best_classical_strategies(copy_f, chsh_g)
#                 print("Best classical :", score)
#                 print("----------------")
