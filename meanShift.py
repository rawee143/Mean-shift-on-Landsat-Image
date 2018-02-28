import sklearn.cluster
from sklearn.cluster import MeanShift, KMeans
from scipy import sparse
from time import time
from sklearn.metrics import calinski_harabaz_score,silhouette_score

def runMeanShift(data,samples,bandwidth):
	ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding = True)
	ms.fit(samples)
	mlabels = ms.labels_
	predY1=ms.predict(data)
	print("Analysing MeanShift:")
	print("Calinski_harabaz_score is %0.3fs" % calinski_harabaz_score(data,predY1))
	sparce_sample =sparse.csr_matrix(data)
	print("Silhouette_score is %0.3fs" % silhouette_score(sparce_sample,predY1, metric='manhattan'))
	runKMeans(data,len(ms.cluster_centers_),samples)
	
def runKMeans(data,n_clusters,samples):
	kclf = sklearn.cluster.KMeans(n_clusters=n_clusters,n_jobs=-1)
	kclf.fit(samples)
	klabels = kclf.labels_ 
	predY= kclf.predict(data)
	print("Analysing KMeans:")
	print("Calinski_harabaz_score is %0.3fs" % calinski_harabaz_score(data,predY))
	sparce_sample =sparse.csr_matrix(data)
	print("Silhouette_score is %0.3fs" % silhouette_score(sparce_sample,predY, metric='manhattan'))