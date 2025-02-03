import numpy as np
import random

def create_matrix(sequence):
    relations = [relation.strip() for relation in sequence.split(",")]
    relations = [relation.split("->") for relation in relations]
    letters = set()
    for relation in relations:
        for letter in relation:
            letters.add(letter.replace("¬", ""))
    letters = list(letters)
    letters.sort()
    Letters = np.copy(letters)
    for i in range(len(letters)):
        Letters = np.append(Letters, "¬" + letters[i])
    letter_to_index = {Letter: i for i, Letter in enumerate(Letters)}
    matrix = [[0 for _ in range(len(Letters))] for _ in range(len(Letters))]

    for relation in relations:
        i = letter_to_index[relation[0]]
        j = letter_to_index[relation[1]]
        matrix[i][j] = 1
    npmatrix = np.array(matrix)
    return npmatrix, Letters

def extend_negation(matrix):
    N = len(matrix) // 2
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (matrix[i][j] == 1) and (i>N-1 or j>N-1):
                matrix[(j+N) % (2*N)][(i+N) % (2*N)] = 1
    return matrix


def check_matrix(matrix):
    for row in matrix:
        if all(element == 0 for element in row):
            return True
    return False


def extend_matrix(p_matrix):
    matrix = np.copy(p_matrix)
    extended_row = [False] * len(matrix)
    for i in range(len(matrix)):
        extend_row(i, extended_row, matrix)
    extended_matrix = np.where(matrix > 0, 1, 0)
    return extended_matrix

def extend_row(row, flag, matrix):
    if flag[row] == 0:
        flag[row] = True
        for i in range(len(matrix)):
            if matrix[row][i] == 1:
                matrix[row] = matrix[row] + extend_row(i, flag, matrix)
    return matrix[row]


def matrix_to_string(matrix, letters):
    N = len(matrix) // 2

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i][j] = 0
            if matrix[i][j] == 1:
                matrix[(j + N) % (2 * N)][(i + N) % (2 * N)] = 0

    '''
    for j in range(2):
        max_row = 0
        max_row_index = 0
        for i in range(len(matrix)):
            sum_column = sum([matrix[k][(i + N) % (2 * N)] for k in range(len(matrix))])
            if (sum(matrix[i]) + sum_column) > max_row:
                max_row_index = i
                max_row = sum(matrix[i]) + sum_column
        matrix[max_row_index] = [0] * len(matrix)
        for i in range(len(matrix)):
            matrix[i][(max_row_index + N) % (2 * N)] = 0
    '''
    '''
    count = 0
    for i in range(len(matrix)):
        count += sum(matrix[i])
    if count > 10:
        index = []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j] != 0:
                    index.append((i, j))
        index = random.sample(index, 10)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if (i, j) not in index:
                    matrix[i][j] = 0
    '''

    relations = []
    for i in range(N*2):
        for j in range(N*2):
            if matrix[i][j] == 1:
                relations.append(f"{letters[i]}->{letters[j]}")
    return relations



def Logic_extend(expression_extracted):
    origin_matrix, letters = create_matrix(expression_extracted)
    pro_matrix = extend_negation(origin_matrix)
    new_matrix = extend_matrix(pro_matrix) - pro_matrix

    return matrix_to_string(new_matrix, letters)

