from copy import deepcopy

import numpy as np

from clusters import MicroCluster
from utilities import manhattan_distance

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
	# @param context			A 2xd NumPy matrix where d is the number of
	#							dimensions, containing the minimum (row 0) and
	#							maximum (row 1) value of each dimension.
	#							Passing the default value of None will activate
	#							adaptive normalization.
	# @param t_global			Integer determining the "Period of the
	#							density-based clustering cycle" (page 17).
	# @param uncdim				Integer specifying the number of dimensions
	#							along which microclusters do not have to
	#							overlap to be considered connected.
	# @param kdtree				Set to True to enable faster connected
	#							micro-cluster search via use of a k-d tree.
	def __init__(self, phi, forget_method=None, ltm=False,
		unclass_accepted=True, minimum_mc=False, multi_density=False,
		context=None, t_global=1, uncdim=0, kdtree=False):
		assert phi >= 0 and phi <= 1, "Invalid phi given"
		if phi > 0.5:
			print("Warning: relative size (phi) > 0.5 may yield poor results")

		# Algorithm customization
		self.phi = phi
		self.forget_method = forget_method
		self.ltm = ltm
		self.density_stage = self._density_stage_local if multi_density else \
			self._density_stage_global

		if context is None:
			self.context = None
			self.norm_func = self._adaptive_normalize_and_update
		else: # Append max-min row for reduced calculations
			self.context = np.append(context, (context[1] - context[0]),
				axis=0)
			self.norm_func = self._normalize

		self.t_global = t_global
		self.uncdim = uncdim
		self.common_dims = context.shape[1] - uncdim if context is not None \
			else None # d - uncdim
		self.spatial_search = self._search_kdtree if kdtree else \
			self._search_all_clusters

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
		return self.phi * np.abs(self.context[2])


	# Helper to calculate the current microcluster hyperbox volume.
		# Vol = product of all sizes
	def _get_hyperbox_volume(self):
		return np.prod(self._get_hyperbox_sizes)

	def _get_next_class_id(self):
		temp = self.next_class_id
		self.next_class_id += 1
		return temp

	# Normalization function for use in cases of no context matrix provided;
	# also, must update the hypervolumes of all microclusters.
	def _adaptive_normalize_and_update(self, X):
		diffmin = X - self.context[0]
		diffmax = X - self.context[1]
		diffminbool = diffmin < 0
		diffmaxbool = diffmax > 0
		updatemins = np.any(diffminbool)
		updatemaxs = np.any(diffmaxbool)
		if updatemins:
			self.context[0] = self.context[0] + np.where(diffminbool,
				diffmin, 0)
		if updatemaxs:
			self.context[1] = self.context[1] + np.where(diffmaxbool,
				diffmax, 0)

		# Update differences
		self.context[2] = self.context[1] - self.context[0]

		# Update hyperbox statistics
		self.hyperbox_sizes = self._get_hyperbox_sizes()
		self.hyperbox_volume = self._get_hyperbox_volume()

		# Need to re-normalize all existing microclusters and return normalized
		# point

	# Normalization function for use in case of passed context matrix.
	def _normalize(self, X):
		return ((X - self.context[0])/(self.context[2]))


	# Helper function to determine if a microcluster is reachable from new
	# instance X.
	def _is_reachable(self, microcluster, X):
		diff = np.abs(X - microcluster.center)
		halvedsize = self.hyperbox_sizes / 2
		return np.all(diff < halvedsize)


	# Helper function to find all reachable microclusters in a given list
	def _find_reachables(self, curr_list, X):
		return [uC for uC in curr_list if self._is_reachable(uC, X)]


	# Helper function to determine if two microclusters are connected.
	# Implements definition 3.2 (page 7) - doesn't insist on a particular
	# subset of features, only that a subset could be created to satisfy
	# overlap among (d - uncdim) dimensions.
	def _is_connected(self, microclusterA, microclusterB):
		diff = np.abs(microclusterA.center - microclusterB.center)
		halvedsize = self.hyperbox_sizes / 2
		numoverlapdims = np.sum((diff < halvedsize).astype(np.uint8))
		if numoverlapdims >= self.common_dims: # Check for sufficient overlap
			return True
		else:
			return False


	# Helper function to find all neighbors in the density stage by searching
	# all clusters.
	def _search_all_clusters(self, curruC, clist):
		return [uC for uC in clist if self._is_connected(uC, curruC)]


	# Helper function to find all neighbors in the density stage by using a
	# kdtree.
	def _search_kdtree(self, curruC, clist):
		pass


	# Helper function to manage final cluster snapshot history
	def _update_snapshots(self):
		pass


	# Helper function for finding the best neighbor microcluster for insertion
	# in a list of microclusters by distance, breaking ties based on density.
	def _find_best_neighbor(self, neighbor_list, X):
		closest = []
		min_dist = np.inf

		for uC in neighbor_list:
			dist = manhattan_distance(uC.center, X)
			if dist < min_dist:
				closest = [uC]
			elif dist == min_dist: # Ties
				closest.append(uC)

		# Sort by density to break ties among equally close
		# microclusters
		closest.sort(key = lambda uC : uC.Dk)
		return closest[0]

	# Implements algorithm 1, distance stage (page 6).
	def _distance_stage(self, X, tX, X_class=None):
		# No microclusters ever created
		if len(self.A_list) == 0 and len(self.O_list) == 0 and \
			len(self.long_term_mem) == 0:
			# If doing adaptive normalization, cannot expect valid density
			# after only one insertion.
			volume = self.hyperbox_volume if self.hyperbox_volume is not None \
				else 1
			self.O_list.append(MicroCluster(X, tX, volume, X_class,
				self.forget_method))
		else: # First check A-list
			Reachables = self._find_reachables(self.A_list, X)
			if len(Reachables) != 0: # If there are reachable microclusters
				best_match = self._find_best_neighbor(Reachables, X)
				best_match.insert(X, tX, X_class)
			else: # Then check O-list
				Reachables = self._find_reachables(self.O_list, X)
				if len(Reachables) != 0: # If there are reachable microclusters
					# Find closest uC in Reachables
					best_match = self._find_best_neighbor(Reachables, X)
					best_match.insert(X, tX, X_class)
				else: # Check long term memory
					Reachables = self._find_reachables(self.long_term_mem, X)
					if self.ltm and len(Reachables) != 0:
						best_match = self._find_best_neighbor(Reachables, X)
						resurrected = deepcopy(best_match)
						resurrected.insert(X, tX, X_class)
						self.O_list.append(resurrected)
					else: # Create uC with X info
						self.O_list.append(MicroCluster(X, tX, volume, X_class,
							self.forget_method))

	# Returns average and median density of given list of clusters.
	def _get_avg_med_density(self, clist):
		cla = np.array([uC.Dk for uC in clist], dtype=np.float64)
		return np.mean(cla), np.median(cla)


	# Implements local density analysis stage.
	def _density_stage_local(self, ):
		pass


	# Implements algorithm 2, density stage - global analysis.
	# Returns updated A-list and O-list
	def _density_stage_global(self):
		allclusters = self.A_list + self.O_list # Flatten cluster lists
		g_avg, g_med = self._get_avg_med_density(allclusters)
		DMC = {uC for uC in allclusters if uC.Dk > g_avg and uC.Dk > g_med}
		already_seen = set()
		for uC in DMC:
			if uC not in already_seen:
				already_seen.add(uC)
				if uC.Classk == 'Unclassed':
					label = self._get_next_class_id()
					uC.Classk = label
			Connected_uC = self.spatial_search(uC,
				list(DMC.difference(already_seen))) # Connected AND dense
			for uCneighbor in Connected_uC:
				uCneighbor.Classk = label
				already_seen.add(uCneighbor)
				NewConnected_uC = self.spatial_search(uCneighbor,
					list(DMC.difference(already_seen)))
				for newneighbor in NewConnected_uC:
					Connected_uC.append(newneighbor)
					newneighbor.Classk = label

		## NEED TO INSERT LOGIC FOR FINAL CLUSTERING (INCLUDING MINIMUM_MC
		## LOGIC), SNAPSHOTS AND FILTERING ALLCLUSTERS TO RETURN THE BELOW
		## LISTS - ASSIGNING DENSITY TYPES AND ASSIGNING TO A-LIST OR O-LIST
		## ACCORDINGLY, "DELETING" BY NOT INCLUDING IN THE RETURNED LISTS IF A
		## uC DOES NOT MEET THE LOW-DENSITY THRESHOLD OR MARKING FOR LONG-TERM
		## MEMORY

		## Need to return dense and semi-dense as A-list, outliers as O-list
		return [], []


	# Runs the DyClee algorithm on a set part of a finite dataset.
	# NOT PART OF THE ORIGINAL DYCLEE ALGORITHM. Designed to avoid running the
	# density stage before reasonable mins and maxs have been observed.
	# @param iterations		The number of instances to process.
	def warm_start(self, iterations):
		pass


	# Runs the DyClee algorithm on a finite dataset.
	# @param data		Data matrix where each row is an instance, each
	#					column is for an attribute.
	# @param timecol	Time increments of the data if available; default of 
	#					None disables time behavior. Forget_method should also
	#					be None.
	# @param targetcol	Column of labels.
	def run_dataset(self, data, timecol=None, targetcol=None):
		# MUST INITIALIZE self.common_dims and self.context
		pass


	# Runs the DyClee algorithm on streaming data.
	def run_datastream(self, ):
		# MUST INITIALIZE self.common_dims and self.context
		pass
