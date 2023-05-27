import torch
import torch.nn.functional as F

def soft_cross_entropy(input, target, reduction='mean'):
	nll = -F.log_softmax(input, dim=1)
	bsz = input.shape[0]
	loss = torch.sum(torch.mul(nll, target))
	if reduction == 'mean':
		loss = loss/bsz
	return loss