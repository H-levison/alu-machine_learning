#!/usr/bin/env python3
"""Performs element-wise operations using NumPy"""


def np_elementwise(mat1, mat2):
    """
    Returns a tuple containing the element-wise sum, 
    difference, product, and quotient
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
