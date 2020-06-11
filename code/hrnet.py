from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)

def down(in_planes,out_planes):
	return nn.Sequential(nn.Conv2d(in_planes,out_planes, kernel_size=3, stride=2, padding=1, bias=False),
						 nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
						 nn.ReLU(inplace=True))

def trans(in_planes,out_planes):
	return nn.Sequential(nn.Conv2d(in_planes,out_planes, kernel_size=3, stride=1, padding=1, bias=False),
						 nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
						 nn.ReLU(inplace=True))

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
							   bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion,
								  momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class HighResolutionModule(nn.Module):
	def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
				 num_channels,  multi_scale_output=True):
		super(HighResolutionModule, self).__init__()
		self._check_branches(
			num_branches, blocks, num_blocks, num_inchannels, num_channels)
		
		self.num_inchannels = num_inchannels
		
		self.num_branches = num_branches
		
		self.multi_scale_output = multi_scale_output
		
		self.branches = self._make_branches(
			num_branches, blocks, num_blocks, num_channels)
		self.fuse_layers = self._make_fuse_layers()
		self.relu = nn.ReLU(True)
	
	def _check_branches(self, num_branches, blocks, num_blocks,
						num_inchannels, num_channels):
		if num_branches != len(num_blocks):
			error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
				num_branches, len(num_blocks))
			logger.error(error_msg)
			raise ValueError(error_msg)
		
		if num_branches != len(num_channels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
				num_branches, len(num_channels))
			logger.error(error_msg)
			raise ValueError(error_msg)
		
		if num_branches != len(num_inchannels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
				num_branches, len(num_inchannels))
			logger.error(error_msg)
			raise ValueError(error_msg)
	
	def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
						 stride=1):
		# ---------------------------(1) begin---------------------------- #
		downsample = None
		if stride != 1 or \
				self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(
					self.num_inchannels[branch_index],
					num_channels[branch_index] * block.expansion,
					kernel_size=1, stride=stride, bias=False
				),
				nn.BatchNorm2d(
					num_channels[branch_index] * block.expansion,
					momentum=BN_MOMENTUM
				),
			)
		# ---------------------------(1) end---------------------------- #
		
		# ---------------------------(2) begin---------------------------- #
		layers = []
		layers.append(
			block(
				self.num_inchannels[branch_index],
				num_channels[branch_index],
				stride,
				downsample
			)
		)
		# ---------------------------(2) middle---------------------------- #
		self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
		for i in range(1, num_blocks[branch_index]):
			layers.append(
				block(
					self.num_inchannels[branch_index],
					num_channels[branch_index]
				)
			)
		# ---------------------------(2) end---------------------------- #
		return nn.Sequential(*layers)
	
	def _make_branches(self, num_branches, block, num_blocks, num_channels):
		branches = []
		
		for i in range(num_branches):
			branches.append(
				self._make_one_branch(i, block, num_blocks, num_channels)
			)
		
		return nn.ModuleList(branches)
	
	def _make_fuse_layers(self):
		# ---------------------------(1) begin---------------------------- #
		if self.num_branches == 1:
			return None
		# ---------------------------(1) end---------------------------- #
		
		num_branches = self.num_branches
		num_inchannels = self.num_inchannels
		# ---------------------------(2) begin---------------------------- #
		fuse_layers = []
		for i in range(num_branches if self.multi_scale_output else 1):
			fuse_layer = []
			for j in range(num_branches):
				# ---------------------------(2.1) begin---------------------------- #
				if j > i:
					fuse_layer.append(
						nn.Sequential(
							nn.Conv2d(
								num_inchannels[j],
								num_inchannels[i],
								1, 1, 0, bias=False
							),
							nn.BatchNorm2d(num_inchannels[i]),
							nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
						)
					)
				# ---------------------------(2.1) end---------------------------- #
				
				# ---------------------------(2.2) begin---------------------------- #
				elif j == i:
					fuse_layer.append(None)
				# ---------------------------(2.2) end---------------------------- #
				
				# ---------------------------(2.3) begin---------------------------- #
				else:
					conv3x3s = []
					for k in range(i - j):
						# ---------------------------(2.3.1) begin---------------------------- #
						if k == i - j - 1:
							num_outchannels_conv3x3 = num_inchannels[i]
							conv3x3s.append(
								nn.Sequential(
									nn.Conv2d(
										num_inchannels[j],
										num_outchannels_conv3x3,
										3, 2, 1, bias=False
									),
									nn.BatchNorm2d(num_outchannels_conv3x3)
								)
							)
					
						else:
							num_outchannels_conv3x3 = num_inchannels[j]
							conv3x3s.append(
								nn.Sequential(
									nn.Conv2d(
										num_inchannels[j],
										num_outchannels_conv3x3,
										3, 2, 1, bias=False
									),
									nn.BatchNorm2d(num_outchannels_conv3x3),
									nn.ReLU(True)
								)
							)
					# ---------------------------(2.3.1) end---------------------------- #
					# ---------------------------(2.3) end---------------------------- #
					fuse_layer.append(nn.Sequential(*conv3x3s))
			fuse_layers.append(nn.ModuleList(fuse_layer))
		# ---------------------------(2) end---------------------------- #
		
		return nn.ModuleList(fuse_layers)
	
	def get_num_inchannels(self):
		return self.num_inchannels
	
	def forward(self, x):
		# ---------------------------(1) begin---------------------------- #
		if self.num_branches == 1:
			return [self.branches[0](x[0])]
		# ---------------------------(1) end---------------------------- #
		
		# ---------------------------(2) begin---------------------------- #
		# ---------------------------(2.1) begin---------------------------- #
		for i in range(self.num_branches):
			x[i] = self.branches[i](x[i])
		# ---------------------------(2.1) end---------------------------- #
		
		# ---------------------------(2.2) begin---------------------------- #
		x_fuse = []
		
		for i in range(len(self.fuse_layers)):
			y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
			for j in range(1, self.num_branches):
				if i == j:
					y = y + x[j]
				else:
					y = y + self.fuse_layers[i][j](x[j])
			x_fuse.append(self.relu(y))
		# ---------------------------(2.2) end---------------------------- #
		# ---------------------------(2) end---------------------------- #
		
		return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


BN_MOMENTUM=0.1

class diy_HRNET(nn.Module):
	
	def __init__(self,cfg,**kwargs):
		super(diy_HRNET,self).__init__()
		
		self.base=nn.Sequential(
			nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(64,momentum=BN_MOMENTUM),
			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,bias=False),
			nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True)
		)
		
		
		self.stage1=self._make_bottleneck_block(64,32)
		
		self.stage2_cfg=cfg['STAGE2']
		num_channels=self.stage2_cfg["NUM_CHANNELS"]
		self.transition1=nn.ModuleList([trans(128,16),down(128,32)])
		self.stage2= self._make_stage(self.stage2_cfg, num_channels)
		
		
		self.stage3_cfg=cfg['STAGE3']
		num_channels=self.stage3_cfg["NUM_CHANNELS"]
		self.transition2=nn.ModuleList([None,None,down(32,64)])
		self.stage3= self._make_stage(self.stage3_cfg, num_channels)
	
		self.stage4_cfg = cfg['STAGE4']
		num_channels = self.stage4_cfg["NUM_CHANNELS"]
		self.transition3=nn.ModuleList([None,None,None,down(64,128)])
		self.stage4= self._make_stage(self.stage4_cfg, num_channels)
		
		# Classification Head
		self.incre_modules=self._make_incremental_layers(num_channels)
		self.downsamp_modules=self._make_downsample_layers()
		self.final_layer = self._make_final_layer(2048)
		
		self.classifier = nn.Linear(2048, 10)
		
	

	def _make_bottleneck_block(self,inplanes,planes,stride=1):
		downsample=nn.Sequential(nn.Conv2d(inplanes,planes*4,kernel_size=1, stride=stride, bias=False),
								 nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),)
		_bottleneck_block=Bottleneck(inplanes,planes,stride, downsample)
		return nn.Sequential(_bottleneck_block)
	
	
	
	def _make_stage(self, layer_config, num_inchannels,
					multi_scale_output=True):
		num_branches = layer_config['NUM_BRANCHES']
		num_blocks = layer_config['NUM_BLOCKS']
		num_channels = layer_config['NUM_CHANNELS']
		block = blocks_dict[layer_config['BLOCK']]
		modules = HighResolutionModule(num_branches,
						block,
						num_blocks,
						num_inchannels,
						num_channels,
						multi_scale_output)
		return nn.Sequential(modules)
	
	def _make_incremental_layers(self,inplanes):
		head_channels=[32,64,128,256]
		incre_modules = []
		for i, channels in enumerate(inplanes):
			incre_module = self._make_bottleneck_block(channels,head_channels[i])
			incre_modules.append(incre_module)
		
		return nn.ModuleList(incre_modules)
	
	def _make_downsample_layers(self):
		head_channels=[128,256,512,1024]
		downsamp_modules = []
		for i in range(3):
			downsamp_modules.append(down(head_channels[i],head_channels[i+1]))
		return  nn.ModuleList(downsamp_modules)
		
	def _make_final_layer(self,out_planes):
		return nn.Sequential(nn.Conv2d(1024,out_planes, kernel_size=3, stride=1, padding=1, bias=False),
							 nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
							 nn.ReLU(inplace=True))
	
	def forward(self, x):
		x = self.base(x)
		x = self.stage1(x)
		
		x_list = []
		for i in range(self.stage2_cfg['NUM_BRANCHES']):
			if self.transition1[i] is not None:
				x_list.append(self.transition1[i](x))
			else:
				x_list.append(x)
		y_list = self.stage2(x_list)
		
		x_list = []
		for i in range(self.stage3_cfg['NUM_BRANCHES']):
			if self.transition2[i] is not None:
				x_list.append(self.transition2[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		y_list = self.stage3(x_list)
		
		x_list = []
		for i in range(self.stage4_cfg['NUM_BRANCHES']):
			if self.transition3[i] is not None:
				x_list.append(self.transition3[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		y_list = self.stage4(x_list)
		
		# Classification Head
		y = self.incre_modules[0](y_list[0])
		for i in range(len(self.downsamp_modules)):
			y = self.incre_modules[i + 1](y_list[i + 1]) + \
				self.downsamp_modules[i](y)
		
		y = self.final_layer(y)
		
		if torch._C._get_tracing_state():
			y = y.flatten(start_dim=2).mean(dim=2)
		else:
			y = F.avg_pool2d(y, kernel_size=y.size()
			[2:]).view(y.size(0), -1)
		
		y = self.classifier(y)
		
		return y
	
	def init_weights(self, pretrained='', ):
		logger.info('=> init weights from normal distribution')
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(
					m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	
def get_cls_net(config, **kwargs):
    model = diy_HRNET(config, **kwargs)
    model.init_weights()
    return model

            
