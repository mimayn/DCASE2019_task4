import numpy as np
import torch
from pdb import set_trace as pause 
import torch.nn as nn
import config as cfg
import torch.nn.functional as F
eps =torch.finfo(torch.float32).eps


def TV_loss(att_map,tv_lambda=1):
	return tv_lambda*torch.mean(torch.sum(torch.pow(att_map[:,:,1:,:] - att_map[:,:,:-1,:],2),-2))
	#return tv_lambda*torch.sum(torch.sum(torch.pow(torch.exp(att_map[:,:,1:,:]) - torch.exp(att_map[:,:,:-1,:]),2),-2))


def asynchrony_loss(att_map, asyn_measure = 'mse-against-all', asyn_lambda=1):
	#return syn_lambda*torch.sum(torch.abs(t_att[:,:,:,:,:] - t_att[:,:,:,:,:]))
	
	dims = att_map.size()
	if asyn_measure == 'var':
		mean = torch.mean(att_map,1).unsqueeze(1)
		return asyn_lambda*torch.mean(torch.sum(torch.pow(att_map-mean,2),-2))
	elif asyn_measure == 'mse-all':
		loss=0
		for i in range(dims[1]):
			for j in range(i+1,dims[1]):
				#loss+=torch.mean(torch.sum(torch.pow(
				#	att_map[:,i,:,:,:]- att_map[:,j,:,:,:],2),-2))
				loss+=torch.mean(torch.sum(torch.abs(
					att_map[:,i,:,:]- att_map[:,j,:,:]),-2))			
		return asyn_lambda* loss*2/(dims[1]*(dims[1]-1))
		#return asyn_lambda* loss
	elif asyn_measure == 'mse-vs-orig':
		
		loss=0
		for i in range(1,dims[1]):
			loss+=torch.sum(torch.sum(torch.pow(
					att_map[:,0,:,:]- att_map[:,i,:,:],2),-2))
		return asyn_lambda*loss/(dims[1]-1)				
	

	elif asyn_measure=='entropy':
		
		max_ent =torch.log(cfg.n_perturb*torch.ones(1)+1).cuda()
		att_map = att_map + eps                                   
		softmax = nn.Softmax(dim=1)
		at = softmax(att_map)
		#non-softmax normalization
		#at = att_map/torch.sum(att_map,1).unsqueeze(1)
		point_loss = max_ent + torch.sum(torch.log(at)*at,1)
		return asyn_lambda*torch.mean(torch.sum(point_loss,-2))	
	
	elif asyn_measure == 'CE-vs-orig':
					
		#using the att_map of the original 
		#example as reference against the transformations

		target = att_map[:,0,:,:].unsqueeze(1).expand_as(att_map[:,1:,:,:])  
		return asyn_lambda*F.binary_cross_entropy(att_map[:,1:,:,:],target.detach())*dims[1]/(dims[1]-1)

	elif asyn_measure == 'CE-vs-all':	
		#use each perturbation version as target calculate the loss and average over all 
		
		loss = 0
		for p in range(att_map.shape[1]):
			nonref_indices = list(range(dims[1]))
			nonref_indices.remove(p)
			target = att_map[:,p,:,:].unsqueeze(1).expand_as(att_map[:,nonref_indices,:,:])		
			loss += F.binary_cross_entropy(att_map[:,nonref_indices,:,:],target.detach())
		return asyn_lambda*loss/att_map.shape[1]	
	
	elif asyn_measure =='cross-corr':
		pass

	elif asyn_measure == 'IoU':
		
		intersection = torch.exp(torch.mean(torch.log(att_map),1))	
		union = torch.mean(att_map,1)
		return torch.mean((eps+intersection)/(eps+union))

def binarization_loss(att_map,bin_lambda=1):
	return bin_lambda*torch.mean(torch.sum(att_map*(1-att_map),-2))	


def aux_weak_loss(att_map, target, aux_lambda=1):
	
	att_map=att_map.view(-1,*att_map.shape[2:])
	inactive_coeff = 1
	active_coeff = 1
	eps =torch.finfo(torch.float32).eps
	act =torch.sum(att_map,-2)/att_map.shape[-2]
	n_active = target.sum()
	n_inactive = target.flatten().size()[0]- n_active
	activity_loss = aux_lambda*torch.sum(target*(torch.exp(-20*torch.sqrt(act+eps))))#/(n_active)
	inactivity_loss = aux_lambda*torch.sum((1-target)*(torch.abs(act-target)))#/n_inactive

	if n_active == 0:
		return inactive_coeff*inactivity_loss/n_inactive
	elif n_inactive == 0:
		return active_coeff*activity_loss/n_active
	else:		
		return active_coeff*activity_loss/n_active + inactive_coeff*inactivity_loss/n_inactive


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