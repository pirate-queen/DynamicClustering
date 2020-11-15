import numpy as np

# Citation:
# Nathalie Barbosa Roa, Louise Travé-Massuyès, Victor Hugo Grisales.
# DyClee: Dynamic clustering for tracking evolving environments.
# Pattern Recognition, Elsevier, 2019, 94, pp.162-186.
# 10.1016/j.patcog.2019.05.024 . hal-02135580

# Implements a non-parallelized version of the DyClee algorithm, as described
# in the 2019 paper.
class SerialDyClee:
	# All default parameter values are as described in the paper (pages 15-17).
	# @param phi				The only required parameter: hyperbox
	# 							relative-size. Float in range [0, 1].
	# @param forget_method		Forgetting process function (callable).
	# @param ltm				Boolean to activate/deactivate long term
	#							memory behavior.
	# @param unclass_accepted	Boolean to enable (True) / disable (False)
	#							outlier rejection.
	# @param minimum_mc			Boolean to retain (False) all clusters or
	#							discard (True) clusters with few microclusters.
	# @param multi_density		Boolean to apply global density (True) or local
	#							density (False) analysis.
	# @param context			A 2xd matrix where d is the number of
	#							dimensions, containing the minimum (row 0) and
	#							maximum (row 1) value of each dimension.
	#							Passing the default value of None will activate
	#							adaptive normalization.
	# @param t_global			Integer determining the "Period of the
	#							density-based clustering cycle" (page 17).
	# @param uncdim				Integer specifying the number of dimensions
	#							along which microclusters do not have to
	#							overlap to be considered connected.
	def __init__(self, phi, forget_method=None, ltm=False,
		unclass_accepted=True, minimum_mc=False, multi_density=False,
		context=None, t_global=1, uncdim=0):
		assert phi >= 0 and phi <= 1, "Invalid phi given"
		if phi > 0.5:
			print("Warning: relative size (phi) > 0.5 may yield poor results")

		# Algorithm customization
		self.phi = phi
		self.forget_method = forget_method
		self.ltm = ltm
		self.context = context
		self.norm_func = self._adaptive_normalize if context is None else \
			self._normalize
		self.t_global = t_global
		self.uncdim = uncdim

		# Storage
		self.A_list = [] # Active medium and high density microclusters
		self.O_list = [] # Low density microclusters
		self.long_term_mem = [] # All microclusters once dense, now low density
		self.snapshots = {} # Need to define how this is maintained

		# Other
		self.hyperbox_sizes = self._get_hyperbox_sizes() if context is not \
			None else None
		self.hyperbox_volume = self._get_hyperbox_volume() if context is not \
			None else None
		self.next_class_id = 0

	# Helper to calculate microcluster hyperbox sizes along each dimension.
	# Size = phi * |dmax - dmin|
	def _get_hyperbox_sizes(self):
		return self.phi * np.abs(self.context[1] - self.context[0])


	# Helper to calculate the current microcluster hyperbox volume.
		# Vol = product of all sizes
	def _get_hyperbox_volume(self):
		return np.prod(self._get_hyperbox_sizes)


	# Normalization function for use in cases of no context matrix provided;
	# also, must update the hypervolumes of all microclusters.
	def _adaptive_normalize_and_update(self, X):
		pass


	# Normalization function for use in case of passed context matrix.
	def _normalize(self, X):
		pass


	# Helper function to determine if a microcluster is reachable from new
	# instance X.
	def _is_reachable(self, microcluster, X):
		pass


	# Helper function to determine if two microclusters are connected.
	def _is_connected(self, microclusterA, microclusterB):
		pass


	# Helper function to find all neighbors in the density stage.
	def _search_kdtree(self, ):
		pass


	# Helper function to manage final cluster snapshot history
	def _update_snapshots(self):
		pass


	# Implements algorithm 1, distance stage.
	def _distance_stage(self, ):
		pass


	# Implements algorithm 2, density stage.
	def _density_stage(self, ):
		pass


	# Runs the DyClee algorithm on a finite dataset.
	def run_dataset(self, ):
		pass


	# Runs the DyClee algorithm on streaming data.
	def run_datastream(self, ):
		pass
