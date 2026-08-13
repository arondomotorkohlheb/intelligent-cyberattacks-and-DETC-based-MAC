from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from scipy.linalg import solve_discrete_are
import numpy as np

if TYPE_CHECKING:
    from OTSystem.state_estimator import Estimator
    from OTSystem.plant import LTImodel


class Controller(ABC):
    def __init__(self):
        self.estimator: Estimator
        self.last_control_input = np.array([0])
        self.biggest_control_input = 0
        self.name: str = "controller"
        self.controller_on = True

    def reset(self):
        self.last_control_input = np.array([0])
        self.biggest_control_input = 0
        self.controller_on = True

    @property
    @abstractmethod
    def info(self) -> dict:
        pass
    
    def __str__(self):
        return str(self.info)
    
    @abstractmethod
    def control(self) -> np.ndarray:
        pass

class LQR(Controller):
    def __init__(self, LTImodel: LTImodel, estimator: Estimator, Q = np.diag([0, 0, 1.85, 142]), R = np.diag([400])):
        super().__init__()
        self.LTImodel = LTImodel
        self.Q = Q
        self.R = R
        self.K = self.compute_K()
        self.estimator = estimator
        self.last_control_input = np.array([0])
        self.biggest_control_input = 0
        self.name = "LQR"

    def compute_K(self):
        P = solve_discrete_are(self.LTImodel.Ad, self.LTImodel.Bd, self.Q, self.R)
        return np.linalg.inv(self.R + self.LTImodel.Bd.T @ P @ self.LTImodel.Bd) @ self.LTImodel.Bd.T @ P @ self.LTImodel.Ad

    def control(self):
        if np.abs(self.estimator.state_estimate[3]) < np.pi/2.5 and self.controller_on:
            u = -self.K @ self.estimator.state_estimate
        else:
            u = np.array([0]) # do not apply control if alpha is too large to avoid explosion of numerical processes
            self.controller_on = False
        
        self.last_control_input = u
        self.biggest_control_input = max(self.biggest_control_input, np.abs(u))
        return u
    
    @property
    def info(self):
        return {
            "type": "LQR",
            "K": self.K,
            "Q": self.Q,
            "R": self.R
        }

class MPC(Controller):
    def __init__(self, LTImodel: LTImodel, estimator: Estimator, Q = np.diag([0, 0, 1.85, 142]), R = np.diag([400])):
        super().__init__()
        self.LTImodel = LTImodel
        self.Q = Q
        self.R = R
        self.estimator = estimator
        self.N = 5
        self.last_control_input = np.array([0.])
        # terminal cost P from discrete algebraic Riccati equation
        self.P = solve_discrete_are(self.LTImodel.Ad, self.LTImodel.Bd, self.Q, self.R)
        # small regularization to ensure H is PD
        self._reg = 1e-8
        self.name = "MPC"

    @property
    def info(self):
        return {
            "type": "MPC",
            "Q": self.Q,
            "R": self.R,
            "N": self.N,
            "P": self.P
        }

    def set_horizon(self, N: int):
        self.N = int(N)

    def _build_prediction_matrices(self):
        A = self.LTImodel.Ad
        B = self.LTImodel.Bd
        n = A.shape[0]
        m = B.shape[1]
        N = self.N

        # A_bar: stack of A^1 ... A^N (n*N x n)
        A_pows = []
        Ap = np.eye(n)
        for i in range(1, N+1):
            Ap = Ap @ A
            A_pows.append(Ap.copy())
        A_bar = np.vstack(A_pows)

        # B_bar: lower block triangular (n*N x m*N)
        B_bar = np.zeros((n * N, m * N))
        for row in range(N):
            for col in range(row+1):
                Apow = np.eye(n)
                for _ in range(row-col):
                    Apow = Apow @ A
                B_bar[row*n:(row+1)*n, col*m:(col+1)*m] = Apow @ B

        # Qbar: block diag of Q,...,Q, P (terminal)
        Q_blocks = [self.Q.copy() for _ in range(N)]
        Q_blocks[-1] = self.P
        Qbar = np.block([[Q_blocks[i] if i==j else np.zeros_like(self.Q) for j in range(N)] for i in range(N)])

        # Rbar: block diag of R repeated N
        Rbar = np.block([[self.R if i==j else np.zeros_like(self.R) for j in range(N)] for i in range(N)])

        return A_bar, B_bar, Qbar, Rbar

    def compute_K_sequence(self):
        """Compute finite-horizon LQR gains K_0 .. K_{N-1} by backward Riccati recursion
        starting from terminal cost P_N = P (solution of DARE or provided terminal cost).
        Returns array of shape (N, m, n) where each K_i maps state->input: u_i = -K_i x_i.
        """
        A = self.LTImodel.Ad
        B = self.LTImodel.Bd
        n = A.shape[0]
        m = B.shape[1]
        N = self.N

        P_next = self.P.copy()
        K_seq = [np.zeros((m, n)) for _ in range(N)]
        P_seq = [None] * (N + 1)
        P_seq[N] = P_next

        for k in range(N - 1, -1, -1):
            S = self.R + B.T @ P_next @ B
            Kk = np.linalg.solve(S, B.T @ P_next @ A)
            K_seq[k] = Kk
            # Riccati backward update
            Pk = A.T @ P_next @ A - A.T @ P_next @ B @ Kk + self.Q
            P_seq[k] = Pk
            P_next = Pk

        # return as numpy array ordered K_0 .. K_{N-1}
        return np.stack(K_seq, axis=0), P_seq

    def build_closed_loop_matrices(self, K_seq=None):
        """Build S and M matrices from the sequence of gains K_seq.
        x_{0:N-1} = S x0, u_{0:N-1} = M x0 (note M includes the negative sign: u = M x0 = -K_seq * ...)
        """
        A = self.LTImodel.Ad
        B = self.LTImodel.Bd
        n = A.shape[0]
        m = B.shape[1]
        N = self.N

        if K_seq is None:
            K_seq, _ = self.compute_K_sequence()

        # build S: stack of I, (A-BK0), (A-BK1)(A-BK0), ...
        S_blocks = []
        prod = np.eye(n)
        S_blocks.append(prod.copy())
        for i in range(1, N):
            prod = (A - B @ K_seq[i-1]) @ prod
            S_blocks.append(prod.copy())
        S = np.vstack(S_blocks)

        # build K_diag (mN x nN) block-diagonal of K_i
        K_diag = np.zeros((m * N, n * N))
        for i in range(N):
            K_diag[i*m:(i+1)*m, i*n:(i+1)*n] = K_seq[i]

        # construct block-repeated S to multiply with K_diag: S is nN x n
        # M = - K_diag @ S  => shape mN x n
        # But K_diag is mN x nN and S is nN x n, so build S_tiled as block rows
        # S already matches as nN x n
        M = - K_diag @ S

        return S, M

    def compute_cost(self, x0, K_seq=None):
        """Compute the finite-horizon cost V_N for initial state x0 using
        closed-loop prediction (S, M) and stage/terminal costs.
        """
        x0 = np.asarray(x0).reshape(-1, 1)
        A_bar, B_bar, Qbar, Rbar = self._build_prediction_matrices()
        if K_seq is None:
            K_seq, _ = self.compute_K_sequence()
        S, M = self.build_closed_loop_matrices(K_seq)

        # V = x0' (S' Qbar S + M' Rbar M) x0
        Hx = S.T @ Qbar @ S + M.T @ Rbar @ M
        V = float(x0.T @ Hx @ x0)
        return V

    def control(self):
        """Compute MPC control using finite-horizon LQR gains (closed-loop prediction).
        Returns first input `u0 = -K_0 x0` (clipped by `u_limit` if present).
        """

        if np.abs(self.estimator.state_estimate[3]) < np.pi/2.5 and self.controller_on:
            x0 = self.estimator.state_estimate
            K_seq, _ = self.compute_K_sequence()
            K0 = K_seq[0]
            u = - (K0 @ x0)
        else:
            u = np.array([0]) # do not apply control if alpha is too large to avoid explosion of numerical processes
            self.controller_on = False
        
        self.last_control_input = u
        self.biggest_control_input = max(self.biggest_control_input, np.abs(u))
        return u