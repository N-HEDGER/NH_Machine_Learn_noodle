import os
import imageio
import matplotlib
import sklearn
from matplotlib import pyplot as plt


rootpath='/Users/nickhedger/Documents/Machine_Learn/SYNS'
SC_prefix='SC'
im_prexix='Im'
numscene=75



def getimages(rootpath,numscene,numims):
	a=[]
	for scene in range(1,numscene+1):
		path=os.path.join(rootpath,SC_prefix)+str(scene)
		print (scene)
		path2=os.path.join(path,'Im1.tif')
		a.append(imageio.imread(path2))
	return(a)
		





