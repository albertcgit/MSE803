import numpy as np

A = np.array([[1,  2,  3],
              [5,  6,  7],
              [10, 0, 11]])

B = np.array([[5, 3],
              [9, 1]])

# Manual cross-correlation using numpy (no flip)
output_rows = A.shape[0] - B.shape[0] + 1
output_cols = A.shape[1] - B.shape[1] + 1
C = np.zeros((output_rows, output_cols), dtype=int)

for i in range(output_rows):
    for j in range(output_cols):
        C[i, j] = np.sum(A[i:i+B.shape[0], j:j+B.shape[1]] * B)

print("Matrix A:")
print(A)
print("\nKernel B:")
print(B)
print("\nConvolution Result C (cross-correlation, valid mode):")
print(C)