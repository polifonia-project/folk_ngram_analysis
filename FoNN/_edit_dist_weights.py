"""Utility script to set up and read custom edit distance penalty matrices for similarity_search_dev.py."""


import numpy as np


# list all ascii characters
index = [chr(i) for i in range(128)]
# create default penalty matrix (square matrix filled with twos),
# to use for edit distance insertion and deletion penalties:

insertion = np.ones(128, dtype=np.float64) + 1

# New: weight insertion by consonance
scale_degrees = (str(i) for i in range(1, 8))
scale_degrees_ascii = [ord(i) for i in scale_degrees]
insertion[scale_degrees_ascii] = [0, 2, 1, 0.5, 0.5, 1, 2]

deletion = insertion
# load custom matrix for edit distance substitution and rotation penalties
substitution_matrix_filename = '_diatonic_substitution_penalty_matrix.csv'
substitution_matrix_dir = '../FoNN'
sub_path = substitution_matrix_dir + '/' + substitution_matrix_filename
sub_matrix = np.loadtxt(sub_path, delimiter=",", dtype=float)
substitution = rotation = np.array(sub_matrix, dtype=float)

# optional print calls to check outputs:
# print("Insertion matrix")
# print(insertion)
# print("Substitution matrix")
# print(substitution)
# print("Deletion matrix")
# print(deletion)
