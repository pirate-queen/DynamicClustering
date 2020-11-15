import numpy as np

# Citation:
# Nathalie Barbosa Roa, Louise Travé-Massuyès, Victor Hugo Grisales.
# DyClee: Dynamic clustering for tracking evolving environments.
# Pattern Recognition, Elsevier, 2019, 94, pp.162-186.
# 10.1016/j.patcog.2019.05.024 . hal-02135580

# Implements the mu-cluster or micro-cluster as described in the 2019 DyClee
# paper.
class MicroCluster:
	# MicroCluster only ever created with a new instance that does not belong
	# to another microcluster.
	# @param X					Initial data instance (NumPy array).
	# @param tX					Time stamp of X.
	# @param V					Current hyperbox volume.
	# @param X_class			Class of instance X, if known.
	# @param decay_function		A decay function (callable) to apply for
	# 							the forgetting process.
	def __init__(self, X, tX, V, X_class=None, decay_function=None):
		# Feature Vector - page 5
		self.nk = 1
		self.LSk = X
		self.SSk = np.power(X, 2)
		self.tlk = self.tsk = tX
		self.Dk = 1 / V
		self.Classk = X_class

		# Calculated statistics
		self.center = X
		self.variance = np.zeros(X.shape, dtype=np.float64)
		self.density_type = None
		self.was_dense = False

		# Function for forgetting process
		self.decay_function = lambda t, t2 : 1 if decay_function is None else \
			decay_function

	# Method to update MicroCluster with new instance X
	# Parameter descriptions can be found in __init__ comments.
	def insert(self, X, tX, X_class=None):
		pass

	# Method to update density upon hyperbox volume change, as in adaptive
	# normalization.
	def update_density(self, V):
		self.Dk = self.nk / V

	# Setter for density_type
	def set_density_type(self, density_type):
		self.density_type = density_type
