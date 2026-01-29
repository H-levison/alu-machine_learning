#!/usr/bin/env python3
"""Adds two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Returns a new matrix representing the sum of two 2D matrices"""
    # Check if number of rows match
    if len(mat1) != len(mat2):
        return None
    
    # Check if number of columns match (assuming uniform shape)
    if len(mat1[0]) != len(mat2[0]):
        return None

    # Use nested list comprehension to create the new matrix
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] 
            for i in range(len(mat1))]
