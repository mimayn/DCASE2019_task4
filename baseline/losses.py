import numpy as np
import torch
from pdb import set_trace as pause 
#import matplotlib.pyplot as plt

def TV_loss(att_map,tv_lambda=1):
	return tv_lambda*torch.mean(torch.abs(att_map[:,:,:,1:,:] - att_map[:,:,:,:-1,:]))


def asynchrony_loss(att_map, asyn_measure = 'mse-against-all', asyn_lambda=1):
	#return syn_lambda*torch.sum(torch.abs(t_att[:,:,:,:,:] - t_att[:,:,:,:,:]))
	#pause()
	dims = att_map.size()
	if asyn_measure == 'var':
		mean = torch.mean(att_map,1)
		return asyn_lambda*torch.mean(torch.pow(att_map-mean,2))
	elif asyn_measure == 'mse-all':
		loss=0
		for i in range(dims[1]):
			for j in range(i+1,dims[1]):
				loss+=torch.mean(torch.pow(
					att_map[:,i,:,:,:]- att_map[:,j,:,:,:],2))
		#return asyn_lambda* loss*2/(dims[1]*(dims[1]-1))
		return asyn_lambda* loss
	elif asyn_measure == 'mse-vs-orig':
		loss=torch.zeros(1).cuda()
		for i in range(1,dims[1]):
			loss+=torch.mean(torch.pow(
					att_map[:,0,:,:,:]- att_map[:,i,:,:,:],2))
		return asyn_lambda*loss#/(dims[1])				
	
			

def binarization_loss(att_map,bin_lambda=1):
	return bin_lambda*torch.mean(att_map*(1-att_map))	





# if __name__ == '__main__':
# 	#x = torch.randn(4,6,1,108,10)
# 	#x = torch.randn(5,10)
# 	x = torch.empty(5,10)
# 	x[:3,:] = torch.randn(3,10)
# 	x[3,:] = x[0,:]
# 	x[4,:] = x[1,:]
# 	for i in range(x.size()[0]):
# 		plt.plot(x[i,:].numpy())

# 	plt.show()	
# 	print(asynchrony_loss(x,1))