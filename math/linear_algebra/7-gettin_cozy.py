#!/usr/bin/env python3
"""Concatenates two 2D matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Returns a new matrix concatenated along axis 0 or 1"""
    if axis == 0:
        # Check if they have the same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Return a new list containing copies of rows from both
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # Check if they have the same number of rows
        if len(mat1) != len(mat2):
            return None
        # Combine rows element-wise
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
