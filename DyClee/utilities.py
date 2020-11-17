import numpy as np

# Citation:
# Nathalie Barbosa Roa, Louise Travé-Massuyès, Victor Hugo Grisales.
# DyClee: Dynamic clustering for tracking evolving environments.
# Pattern Recognition, Elsevier, 2019, 94, pp.162-186.
# 10.1016/j.patcog.2019.05.024 . hal-02135580

# Decay functions
# All functions take metaparameters upon initialization, but expect
# current time (t) and last time (tlk) at runtime

# Implements linear decay (page 12)
# tw_zero = time func takes to go from 1 to 0
# m = function slope
def linear_decay(tw_zero, m):
	def ld(t, tlk):
		if t - tlk > tw_zero:
			return 0
		else:
			return 1-m*(t-tlk)
	return ld

def trapezoidal_decay(t, tlk):
	pass


def zshape_decay(t, tlk):
	pass


def exponential_decay(t, tlk):
	pass


def halflife_decay(t, tlk):
	pass


def sigmoidal_decay(t, tlk):
	pass


# Distance

def manhattan_distance(a, b):
	return np.sum(np.abs(a - b))

