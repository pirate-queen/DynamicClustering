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


# ta = no forgetting time
def trapezoidal_decay(tw_zero, m, ta):
	def td(t, tlk):
		diff = t - tlk
		if diff > tw_zero:
			return 0
		elif diff <= tw_zero and ta <= diff:
			return (m-t)/(m-ta)
		elif diff <= ta:
			return 1
	return td


def zshape_decay(ta, tw_zero):
	def zsd(t, tlk):
		diff = t - tlk
		if t <= ta:
			return 1
		elif ta <= diff and diff <= (ta+tw_zero)/2:
			return 1-2*(t-ta)/(tw_zero-ta)
		elif (ta+tw_zero)/2 <= diff and diff <= tw_zero:
			return 2*(t-tw_zero)/(tw_zero-ta)
		elif diff > tw_zero:
			return 0
	return zsd


def exponential_decay(lam):
	def ed(t, tlk):
		return np.power(np.e, -lam*(t-tlk))
	return ed


def halflife_decay(B, lam):
	def hld(t, tlk):
		return np.power(B, -lam*(t-tlk))
	return hld


def sigmoidal_decay(a, c):
	def sd(t, tlk):
		return 1/(1+np.power(np.e, -a*(t-c)))
	return sd


# Distance

def manhattan_distance(a, b):
	return np.sum(np.abs(a - b))

