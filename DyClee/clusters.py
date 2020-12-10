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
		self.Classk = X_class if X_class is not None else "Unclassed"

		# Calculated statistics
		self.center = X
		self.variance = np.zeros(X.shape, dtype=np.float64)
		self.density_type = "Outlier"
		self.was_dense = False

		# Function for forgetting process
		if decay_function is None:
			self.decay_function = lambda t, t2 : 1
		else:
			self.decay_function = decay_function


	# Helper function to calculate center based on feature vector
	def get_center(self):
		return self.LSk / self.nk


	# Helper function to calculate variance based on feature vector
	def get_variance(self):
		return (self.SSk / self.nk) - np.power((self.LSk / self.nk), 2)


	# Method to update MicroCluster with new instance X
	# Parameter descriptions can be found in __init__ comments.
	def insert(self, X, tX, X_class=None):
		self.nk += 1 # Sample count
		self.LSk += X # Linear sum
		self.SSk += np.power(X,2) # Squared sum
		self.tlk = tX # Update last assignment time
		self.center = self.get_center()
		self.variance = self.get_variance()
		if X_class is not None and self.Classk is None:
			self.Classk = X_class # Update class


	# Method for all MicroClusters to be updated upon time increment
	# @param tX		Current time.
	def update_cluster(self, tX):
		decay_factor = self.decay_function(tX, self.tlk)
		self.nk = (self.nk * decay_factor)
		self.LSk = (self.LSk * decay_factor)
		self.SSk = (self.SSk * decay_factor)


	# Method to update density upon hyperbox volume change, as in adaptive
	# normalization.
	# @param V		Current hyperbox volume.
	def update_density(self, V):
		self.Dk = self.nk / V


	# Setter for density_type
	def set_density_type(self, density_type):
		self.density_type = density_type
		if not self.was_dense and density_type == "Dense":
			self.was_dense = True

class FinalCluster:
	def __init__(self, label, center, density, max_distance):
		self.label = label
		self.center = center
		self.density = density
		self.max_distance = max_distance
