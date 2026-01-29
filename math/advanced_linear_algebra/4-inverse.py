#!/usr/bin/env python3
"""Calculates the inverse of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a square matrix"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def inverse(matrix):
    """Calculates the inverse of a non-singular square matrix"""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # 1. Check if matrix is singular
    det = determinant(matrix)
    if det == 0:
        return None

    # Base case for 1x1 matrix: inverse is 1/element
    if n == 1:
        return [[1 / matrix[0][0]]]

    # 2. Calculate the Cofactor Matrix
    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minor_val = determinant(sub_matrix)
            row_cofactors.append(((-1) ** (i + j)) * minor_val)
        cofactor_matrix.append(row_cofactors)

    # 3. Transpose to get Adjugate and Multiply by 1/det
    inv_matrix = []
    for j in range(n):
        new_row = []
        for i in range(n):
            # Divide each element of the adjugate by the determinant
            new_row.append(cofactor_matrix[i][j] / det)
        inv_matrix.append(new_row)

    return inv_matrix
