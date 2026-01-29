#!/usr/bin/env python3
"""Calculates the determinant of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a square matrix"""
    if matrix == [[]]:
        return 1
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
        
    n = len(matrix)
    if n == 0:
        return 1
    
    # Check if the matrix is square
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][2 - 1]

    # Recursive step for nxn matrix using cofactor expansion
    det = 0
    for j in range(n):
        # Create submatrix by removing the first row and the j-th column
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # Alternate signs (+ - + -) using (-1)**j
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    
    return det
