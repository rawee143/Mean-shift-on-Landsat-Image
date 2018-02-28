import numpy as np
import rasterio

def loadImage():
	img = rasterio.open('a.tif')
	h = img.height
	w = img.width
	r,g,b,x = img.read()
	truth_lbl= b.reshape(-1)
	data = np.dstack((r,g,b))
	return (data,h,w)