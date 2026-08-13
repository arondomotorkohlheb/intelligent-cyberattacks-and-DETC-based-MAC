from __future__ import annotations
from typing import TYPE_CHECKING

from scipy.signal import place_poles
import numpy as np
from filterpy.kalman import KalmanFilter
from abc import ABC, abstractmethod

if TYPE_CHECKING:
	from OTSystem.plant import LTImodel

class Estimator(ABC):
	def __init__(self, LTI: LTImodel):
		self.x_hat = np.zeros(LTI.Ad.shape[0])
		self.LTI = LTI
		self.elapsed_time = self.LTI.ts
		self.name: str = "estimator"

	def reset(self, x_hat0 = None):
		self.x_hat = np.zeros(self.LTI.Ad.shape[0]) if x_hat0 is None else x_hat0

	@property
	def state_estimate(self):
		return self.x_hat.copy()

	@abstractmethod
	def update(self, u, y = None):
		pass

	@property
	@abstractmethod
	def info(self) -> dict:
		pass

	def __str__(self):
		return str(self.info)

class Observer(Estimator):
	"""Discrete-time Luenberger-style state observer."""
	
	def __init__(self, LTI: LTImodel) -> None:
		super().__init__(LTI)
		self.Ad = LTI.Ad
		self.Bd = LTI.Bd
		self.C = LTI.C
		self.L = self.setL()
		self.eig_vals: np.ndarray = np.linalg.eigvals(self.Ad - self.L @ self.C)
		self.name: str = "Observer"
    
	def setL(self):
		observer_poles = np.array([0.1, 0.2, -0.1, -0.2]) # type: ignore
		L = place_poles(self.Ad.T, self.C.T, observer_poles).gain_matrix.T # type: ignore
		return L

	def update(self, u, y = None):
		if y is None:
			y = self.LTI.C @ self.x_hat
		self.x_hat = self.Ad @ self.x_hat + self.Bd @ u + self.L @ (y - self.C @ self.x_hat)

	@property
	def info(self) -> dict:
		return {
			"estimator_type": "Observer",
			"Ad": self.Ad,
			"Bd": self.Bd,
			"C": self.C,
			"L": self.L,
			"observer eigenvalues": self.eig_vals
		}

	
class KalmanEstimator(Estimator):
	def __init__(self, LTI: LTImodel, x_hat0 = None, Q = None, P = None) -> None:
		super().__init__(LTI)
		self.x_hat = np.zeros(LTI.Ad.shape[0]) if x_hat0 is None else x_hat0
		self.kalman_filter = KalmanFilter(dim_x=LTI.Ad.shape[0], dim_z=LTI.C.shape[0])
		self.LTI = LTI
		self.kalman_filter.F = LTI.Ad
		self.kalman_filter.B = LTI.Bd # type: ignore
		self.kalman_filter.H = LTI.C
		self.kalman_filter.x = self.x_hat
		if Q is not None:
			self.kalman_filter.Q = Q
		else:
			self.kalman_filter.Q = np.diag([0.0001, 0.0001, 0.01, 0.1])
		if P is not None:
			self.kalman_filter.P = P
		else:
			self.kalman_filter.P *= 0.003

		self.kalman_filter.R *= 0.1

		self.P0 = self.kalman_filter.P.copy()
		
		self.name: str = "KalmanEstimator"

	def update(self, u, y = None):
		if y is None:
			y = self.LTI.C @ self.x_hat
		self.kalman_filter.predict(u)
		self.kalman_filter.update(y)
		self.x_hat = self.kalman_filter.x.copy()


	def reset(self, x_hat0=None, P0=None):
		super().reset(x_hat0)

		self.kalman_filter.x = self.x_hat.copy()

		if P0 is not None:
			self.kalman_filter.P = P0.copy()
		else:
			self.kalman_filter.P = self.P0.copy()
		
	@property
	def info(self) -> dict:
		return {
			"estimator_type": "KalmanEstimator",
			"Ad": self.LTI.Ad,
			"Bd": self.LTI.Bd,
			"C": self.LTI.C,
			"Q": self.kalman_filter.Q,
			"P": self.kalman_filter.P,
			"R": self.kalman_filter.R
		}

if __name__ == "__main__":
	pass

		