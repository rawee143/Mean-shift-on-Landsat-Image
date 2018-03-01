#import packages
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from time import time
import loadFile as load

t0= time()
data,h,w = load.loadTifImage()
plt.figure(figsize=(12, 12))
plt.imshow(data)
plt.show()

samples = data.reshape(-1,3) 
print(samples.shape)

#sampling small portion of entire data

from sklearn.utils import shuffle
image_array_sample = shuffle(samples, random_state=0)[:10000]

print("Evaluating Bandwidth:")
import evaluateBandwidth as evalBan

scott, silverman, normal,bandwidth = evalBan.meanShift(samples,image_array_sample)

print("Scott_Bandwidth: %0.3fs" %scott)
print("Silverman_Bandwidth: %0.3fs" %silverman)
print("Normal_Reference_Bandwidth: %0.3fs" %normal)
print("KNN_Bandwidth: %0.3fs" %bandwidth)
bandwidth_set =[scott, silverman, normal,bandwidth]


print("MeanShift Started")
import meanShift as mns
for x in range(len(bandwidth_set)):
	print("----------------------------------------")
	mns.runMeanShift(image_array_sample,image_array_sample,bandwidth_set[x])
	print("----------------------------------------")