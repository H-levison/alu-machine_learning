#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies two 2D matrices and returns a new matrix"""
    # Validation: columns of mat1 must equal rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Determine dimensions
    m = len(mat1)      # rows in mat1
    n = len(mat2)      # columns in mat1 / rows in mat2
    p = len(mat2[0])   # columns in mat2

    # Initialize a new matrix with zeros
    new_matrix = [[0 for _ in range(p)] for _ in range(m)]

    # Perform multiplication (Dot product of rows and columns)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                new_matrix[i][j] += mat1[i][k] * mat2[k][j]

    return new_matrix
