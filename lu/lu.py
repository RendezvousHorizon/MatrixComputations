from argparse import ArgumentParser
import timeit
from typing import Tuple
import copy

import numpy as np


class Solver:

    E = 1e-7

    @staticmethod
    def assert_ax_eq_b(a, x, b):
        assert np.max(np.abs(np.matmul(a, x) - b)) < Solver.E

    @staticmethod
    def assert_a_eq_b(a, b):
        assert np.max(np.abs(a - b)) < Solver.E

    @staticmethod
    def solve_lx_eq_b(L: np.matrix, b: np.ndarray, inplace=False) -> np.ndarray:
        # flops = 2 * [(n - 1) + ... + 1] = n^2 - n
        # b can be a 1d array or a matrix
        old_b = copy.deepcopy(b)
        if not inplace:
            b = copy.deepcopy(b)
        for i in range(min(L.shape)):
            b[i + 1:] -= np.matmul(L[i + 1:, i: i + 1], b[i:i + 1])
        Solver.assert_ax_eq_b(L, b, old_b)
        return b

    @staticmethod
    def solve(A: np.matrix, b: np.ndarray, lu_method: str, block_size: int) -> np.ndarray:
        time = timeit.default_timer()
        fact_result = LUFactorator.lu_factorization(A, lu_method, block_size)
        if len(fact_result) == 3:
            LU, L, U = fact_result
        else:
            LU, L, U, P = fact_result
            b = LUFactorator.permute(P, b)

        lu_time = timeit.default_timer() - time
        m, r, n = *L.shape, U.shape[1]
        
        time = timeit.default_timer()
        # LUx = b -> Ly=b + Ux=y
        y = np.zeros(r)
        for i in range(r):
            y[i] = b[i]
            b[i + 1:] -= LU[i + 1:, i] * y[i]

        x = np.zeros(r)
        for j in range(r - 1, -1, -1):
            x[j] = y[j] / LU[j, j]
            y[:j] -= LU[:j, j] * x[j]
        solve_time = timeit.default_timer() - time 
        print(f'Solve LUx=b time: {solve_time / 1000: .3f}ms')

        return x, round(lu_time / 1e3, 3)


class LUFactorator:

    @staticmethod
    def lu_factorization(A: np.ndarray, lu_method: str, block_size: int=None, inplace=False) -> np.ndarray:
        if lu_method == 'outer_product':
            return LUFactorator.lu_factorization_outer_product(A, inplace)
        if lu_method == 'gaxpy':
            return LUFactorator.lu_factorization_gaxpy(A, inplace)
        if lu_method == 'blocked':
            return LUFactorator.lu_factorization_blocked(A, block_size, inplace)
        if lu_method == 'pivoted':
            return LUFactorator.lu_factorization_pivoted(A, inplace)
        if lu_method == 'blocked_pivoted':
            return LUFactorator.lu_factorization_blocked_pivoted(A, block_size, inplace)
        raise ValueError(lu_method)

    @staticmethod
    def separate_lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = A.shape
        r = min(m, n)
        L = np.zeros([m, r])
        U = np.zeros([r, n])
        for i in range(r):
            L[i, i] = 1
            L[i, :i] = A[i, :i]
            U[i, i:] = A[i, i:]
        for i in range(r, m):
            L[i, :] = A[i, :]
        return L, U

    @staticmethod
    def lu_factorization_outer_product(A: np.ndarray, inplace=False):
        # FLOPs 2 / 3 * n^3
        m, n = A.shape
        r = min(m, n)
        A_old = copy.deepcopy(A)
        if not inplace:
            A = copy.deepcopy(A)
        for k in range(r):
            A[k + 1:, k] /= A[k, k] # FLOPS = (r - k)
            A[k + 1:, k + 1:] -= np.matmul(A[k + 1:, k:k + 1], A[k:k + 1, k + 1:]) # FLOPS = (r - k)^2

        L, U = LUFactorator.separate_lu(A)
        Solver.assert_ax_eq_b(L, U, A_old)

        return A, L, U

    @staticmethod
    def lu_factorization_gaxpy(A: np.matrix, inplace=False):
        old_A = copy.deepcopy(A)
        if not inplace:
            A = copy.deepcopy(A) 
        A[1:, 0] /= A[0, 0]
        for j in range(1, min(A.shape)):
            # solve A[:j,j] = L[:j,:j]U[:j,j] FLOPS j^2
            for i in range(j + 1):
                A[i + 1:j + 1, j] -= A[i + 1:j + 1, i] * A[i, j]
            
            # calculate L[j+1:,j] FLOPs j * (n - j)
            A[j+1:, j] = (A[j+1:, j] - np.matmul(A[j+1:, :j], A[:j, j])) / A[j, j]

        L, U = LUFactorator.separate_lu(A)

        Solver.assert_ax_eq_b(L, U, old_A)

        return A, L, U

    @staticmethod
    def l_inverse(L: np.ndarray) -> np.ndarray:
        n = L.shape[0]
        L_inverse = np.identity(n)
        # TODO 
        pass 

    @staticmethod
    def lu_factorization_blocked(A: np.matrix, block_size: int, inplace=False):
        assert A.shape[0] == A.shape[1]
        R = block_size
        submatrix = lambda A, R, i, j: A[i * R:i * R + R, j * R:j * R + R]

        old_A = copy.deepcopy(A)
        if not inplace:
            A = copy.deepcopy(A) 
        M = A.shape[0] // R
        for km in range(M):
            k_start = km * R
            k_end = k_start + R
            # step 1: lu A[k_start:, k_start:k_end]
            LUFactorator.lu_factorization_outer_product(A[k_start:, k_start:k_end], inplace=True)
            # step 2: solve A[k_start:k_end, k_end:] = L[k_start:k_end, k_start:k_end] * U[k_start:k_end, k_end:]
            Lb = A[k_start:k_end, k_start:k_end]
            Lb_real = LUFactorator.separate_lu(Lb)[0]
            for jm in range(km + 1, M):
                # solve Ab_kmjm = Lb * Ub_kmjm
                Abkmjm = submatrix(A, R, km, jm)
                old_Akmjm = submatrix(old_A, R, km, jm)
                for r in range(R):
                    Abkmjm[:, r] = Solver.solve_lx_eq_b(Lb_real, Abkmjm[:, r], inplace=False)
            # step 3: Ab_imjm -= Lbimkm * Ubkmjm
            for im in range(km + 1, M):
                for jm in range(km + 1, M):
                    Abimjm = submatrix(A, R, im, jm)
                    Abimjm -= np.matmul(submatrix(A, R, im, km), submatrix(A, R, km, jm))
    
        L, U = LUFactorator.separate_lu(A)

        Solver.assert_ax_eq_b(L, U, old_A)

        return A, L, U

    @staticmethod
    def build_pivot_matrix(P):
        n = P.shape[0]
        rv = np.zeros([n, n])
        for i, v in enumerate(P):
            rv[i, v] = 1
        return rv 

    @staticmethod
    def permute(P, M):
        if len(P.shape) == 2:
            return np.matmul(P, M)
        else:
            return M[P]
    @staticmethod
    def lu_factorization_pivoted(A: np.matrix, inplace=False):
        old_A = copy.deepcopy(A)
        if not inplace:
            A = copy.deepcopy(A)
        m, n = A.shape
        r = min(m, n)
        P = np.arange(m)
        for k in range(r):
            # pivoting
            pivot = np.argmax(np.abs(A[k:, k])) + k
            if pivot != k:
                tmp = P[k]
                P[k] = P[pivot]
                P[pivot] = tmp
                tmp = A[k, :].copy()
                A[k, :] = A[pivot, :]
                A[pivot, :] = tmp[:]
            # calculation
            A[k + 1:, k] = A[k + 1:, k] / A[k, k]
            A[k + 1:, k + 1:] -= np.matmul(A[k + 1:, k: k + 1], A[k: k + 1, k + 1:])
        L, U = LUFactorator.separate_lu(A)
        Solver.assert_a_eq_b(LUFactorator.permute(P, old_A), np.matmul(L, U))
        return A, L, U, P

    # LU factorization with partial pivoting: PA = LU
    @staticmethod
    def lu_factorization_blocked_pivoted(A: np.matrix, block_size: int, inplace=False):
        assert A.shape[0] == A.shape[1]

        R = block_size
        N = A.shape[0]
        M = N // R
        
        old_A = copy.deepcopy(A)
        if not inplace:
            A = copy.deepcopy(A) 
            
        # 1) panel LU
        _, L, U11, P = LUFactorator.lu_factorization_pivoted(A[:, :R], inplace=True)
        L11 = L[:R]
        L21 = L[R:]

        # stop 
        if N <= R:
            return A, L, U11, P

        # 2) permute right of panel: [A21'T,A22'T]T = P[A21T,A22T]T
        A[:, R:] = LUFactorator.permute(P, A[:, R:])

        # 3) solve L11U12 = A12'
        A12 = A[:R, R:]
        Solver.solve_lx_eq_b(L11, A12, inplace=True)

        # 4) Update A22' = A22' - L21U12 
        A22 = A[R:, R:]
        U12 = A12
        A22 -= np.matmul(L21, U12)

        # 5) Recursion
        _, _, _, P2 = LUFactorator.lu_factorization_blocked_pivoted(A22, R, inplace=True)

        # merge result
        L21 = A[R:, :R]
        L21[:, :] = LUFactorator.permute(P2, L21)
        L, U = LUFactorator.separate_lu(A)
        P[R:] = LUFactorator.permute(P2, P[R:])

        return A, L, U, P


def main():
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--r', type=int, default=10)
    parser.add_argument('--lu_method', choices=['outer_product', 'gaxpy', 'blocked', 'pivoted', 'blocked_pivoted'], type=str, default='outer_product')
    args = parser.parse_args()

    # initial A, b
    A = np.random.randn(args.n, args.n)
    x_correct = np.random.randn(args.n)
    b = np.matmul(A, x_correct)

    x, dur = Solver.solve(A, b, args.lu_method, args.r)

    Solver.assert_a_eq_b(x, x_correct)

    block_size_str = f' r={args.r}' if args.lu_method == 'blocked' else ''
    print(f'N={args.n}, {args.lu_method}{block_size_str} time={dur}ms')


if __name__ == '__main__':
    main()