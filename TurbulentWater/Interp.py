import numpy as np
import torch
import torch.nn.functional

from globals import cuda_if_available


def interp1(f,i):
	# f is the signal to be interpolated with size [num_batch, channels, length]
	# i are the indicies into f with size [num_batch, new_height, new_width]
	# returns a new signal of size [num_batch, channels, new_height, new_width]
	f = f.transpose(1,2).contiguous()
	num_batch, length, channels = f.size()
	new_size = (i.size()+(channels,))
	f_flat = f.view(-1,channels)
	i   = i.clamp(0,length-1)
	i0  = i.floor()
	i0_ = i0.long()
	i1_ = (i0_+1).clamp(0,length-1)
	batch_ind = torch.arange(0,num_batch).long().view(-1,1,1).expand_as(i0)
	if f.is_cuda:
		batch_ind = batch_ind.cuda()
	idx0 = batch_ind*length + i0_
	idx1 = batch_ind*length + i1_
	f0 = f_flat.index_select(0,idx0.view(-1)).view(*new_size)
	f1 = f_flat.index_select(0,idx1.view(-1)).view(*new_size)
	di = (i-i0).unsqueeze(3).expand_as(f0)
	f   = f0*(1-di) + f1*di
	return f.transpose(2,3).transpose(1,2).contiguous()


def interp2(f,i,j):
	# f is the image to be interpolated with size [num_batch, channels, height, width]
	# i,j are grids of indicies into f with size [num_batch, new_height, new_width]
	# returns a new image of size [num_batch, channels, new_height, new_width]
	f = f.transpose(1,2).transpose(2,3).contiguous()
	num_batch, height, width, channels = f.size()
	new_size = (i.size()+(channels,))
	f_flat = f.view(-1,channels)
	i   = i.clamp(0,height-1)
	j   = j.clamp(0,width-1)
	i0  = i.floor()
	j0  = j.floor()
	i0_ = i0.long()
	j0_ = j0.long()
	i1_ = (i0_+1).clamp(0,height-1)
	j1_ = (j0_+1).clamp(0,width-1)
	batch_ind = torch.arange(0,num_batch).long().view(-1,1,1).expand_as(i0)
	if f.is_cuda:
		batch_ind = batch_ind.cuda()
	idx00 = batch_ind*width*height + i0_*width + j0_
	idx01 = batch_ind*width*height + i0_*width + j1_
	idx10 = batch_ind*width*height + i1_*width + j0_
	idx11 = batch_ind*width*height + i1_*width + j1_
	f00 = f_flat.index_select(0,idx00.view(-1)).view(*new_size)
	f01 = f_flat.index_select(0,idx01.view(-1)).view(*new_size)
	f10 = f_flat.index_select(0,idx10.view(-1)).view(*new_size)
	f11 = f_flat.index_select(0,idx11.view(-1)).view(*new_size)
	di = (i-i0).unsqueeze(3).expand_as(f00)
	dj = (j-j0).unsqueeze(3).expand_as(f00)
	f0  = f00*(1-dj) + f01*dj
	f1  = f10*(1-dj) + f11*dj
	f   = f0*(1-di) + f1*di
	return f.transpose(2,3).transpose(1,2).contiguous()


def warp(im,di,dj):
	# f is the image to be interpolated with size [num_batch, channels, height, width]
	# di,dj are grids of index offsets into f with size [num_batch, height, width]
	i,j = np.meshgrid(np.arange(di.size()[1], dtype='float32'), np.arange(di.size()[2], dtype='float32'), indexing='ij')
	i   = torch.from_numpy(i).unsqueeze(0).expand_as(im[:,0,:,:]).float()
	j   = torch.from_numpy(j).unsqueeze(0).expand_as(im[:,0,:,:]).float()
	if im.is_cuda:
		i,j=i.cuda(), j.cuda()
	return interp2(im, i+di, j+dj)


def inverse_warp(im, di, dj):
	"""
	Get observation from original/predicted image

	:param im: [num_batch, channels, height, width]
	:param di: [num_batch, height, width]
	:param dj: [num_batch, height, width]
	:return: [num_batch, channels, height, width]
	"""

	# TODO: implement "splatting" griddata
	num_batch, channels, height, width = im.shape
	i, j = np.meshgrid(
		np.arange(height),
		np.arange(width),
		indexing='ij')
	i = torch.from_numpy(i).unsqueeze(0).expand_as(im[:, 0, :, :])
	j = torch.from_numpy(j).unsqueeze(0).expand_as(im[:, 0, :, :])

	if im.is_cuda:
		i = i.cuda()
		j = j.cuda()

	ii = (i.float() + di).round().clamp(0, height - 1).long()
	jj = (j.float() + dj).round().clamp(0, width - 1).long()

	jm = torch.zeros_like(im)
	jm[:, :, ii, jj] = im[:, :, i, j]

	return jm


# DEPRECATED
def get_inverse_warp(di, dj):
	# TODO: implement "splatting" griddata
	num_batch, height, width = di.shape
	i, j = np.meshgrid(
		np.arange(height),
		np.arange(width),
		indexing='ij')
	i = torch.from_numpy(i).unsqueeze(0).expand_as(di)
	j = torch.from_numpy(j).unsqueeze(0).expand_as(dj)

	if di.is_cuda:
		i = i.cuda()
		j = j.cuda()

	ii = (i.float() + di).round().clamp(0, height - 1).long()
	jj = (j.float() + dj).round().clamp(0, width - 1).long()

	ddi = torch.zeros_like(di)
	ddi[:, ii, jj] -= di[:, i, j]
	ddj = torch.zeros_like(dj)
	ddj[:, ii, jj] -= dj[:, i, j]

	return ddi, ddj


HORIZONTAL_DERIVATIVE = cuda_if_available(torch.FloatTensor([[[[-1, 0, 1]]]]))
VERTICAL_DERIVATIVE = cuda_if_available(torch.FloatTensor([[[[-1, ], [0, ], [1, ]]]]))


def gradient(im):
	di = torch.nn.functional.conv2d(
		im.unsqueeze(1),
		VERTICAL_DERIVATIVE,
		padding=(1, 0),
	).squeeze(1)
	dj = torch.nn.functional.conv2d(
		im.unsqueeze(1),
		HORIZONTAL_DERIVATIVE,
		padding=(0, 1),
	).squeeze(1)

	return di, dj
