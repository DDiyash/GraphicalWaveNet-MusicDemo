import numpy as np

def adjacency(N=128):
    A = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            A[i, i - 1] = 1
        if i < N - 1:
            A[i, i + 1] = 1
    return A
