import sklearn.cluster
import nonparametric_bandwidth as nb
import kernels
from sklearn.cluster import estimate_bandwidth

def meanShift(data,samples):
	xample = samples

	scott_bandwidth = nb.select_bandwidth(xample, bw='scott',kernel=None)
	silverman_bandwidth = nb.select_bandwidth(xample, bw='silverman',kernel=None)
	normal_bandwidth = nb.select_bandwidth(xample, bw='normal_reference',kernel=kernels.Gaussian)
	bandwidth = estimate_bandwidth(data, n_samples=len(xample), quantile=0.3 ,n_jobs=-1)

	return (scott_bandwidth[1], silverman_bandwidth[1],normal_bandwidth[1],bandwidth)
