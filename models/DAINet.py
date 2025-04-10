# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable , Function

from layers import *
from data.config import cfg
from layers.modules import L2Norm  # 添加在文件顶部

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Interpolate( nn.Module ) :
	# 插值的方法对张量进行上采样或下采样
	def __init__( self , scale_factor ) :
		super( Interpolate , self ).__init__()
		self.scale_factor = scale_factor
	
	def forward( self , x ) :
		x = nn.functional.interpolate( x , scale_factor = self.scale_factor , mode = 'nearest' )
		return x


class FEM( nn.Module ) :
	"""docstring for FEM"""
	
	def __init__( self , in_planes ) :
		super( FEM , self ).__init__()
		inter_planes = in_planes // 3
		inter_planes1 = in_planes - 2 * inter_planes
		self.branch1 = nn.Conv2d( in_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 )
		
		self.branch2 = nn.Sequential( nn.Conv2d( in_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) , nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) )
		self.branch3 = nn.Sequential( nn.Conv2d( in_planes , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) , nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes1 , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) , nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes1 , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) )
	
	def forward( self , x ) :
		x1 = self.branch1( x )
		x2 = self.branch2( x )
		x3 = self.branch3( x )
		out = torch.cat( (x1 , x2 , x3) , dim = 1 )
		out = F.relu( out , inplace = True )
		return out


# 拉普拉斯金字塔
class Lap_Pyramid_Conv( nn.Module ) :
	def __init__( self , num_high = 1 , kernel_size = 5 , channels = 3 ) :
		super().__init__()
		
		self.num_high = num_high
		self.kernel = self.gauss_kernel( kernel_size , channels )
	
	def gauss_kernel( self , kernel_size , channels ) :
		kernel = cv2.getGaussianKernel( kernel_size , 0 ).dot( cv2.getGaussianKernel( kernel_size , 0 ).T )
		kernel = torch.FloatTensor( kernel ).unsqueeze( 0 ).repeat( channels , 1 , 1 , 1 )
		kernel = torch.nn.Parameter( data = kernel , requires_grad = False )
		return kernel
	
	def conv_gauss( self , x , kernel ) :
		n_channels , _ , kw , kh = kernel.shape
		x = torch.nn.functional.pad( x , (kw // 2 , kh // 2 , kw // 2 , kh // 2) ,
		                             mode = 'reflect' )  # replicate    # reflect
		x = torch.nn.functional.conv2d( x , kernel , groups = n_channels )
		return x
	
	def downsample( self , x ) :
		return x[ : , : , : :2 , : :2 ]
	
	def pyramid_down( self , x ) :
		return self.downsample( self.conv_gauss( x , self.kernel ) )
	
	def upsample( self , x ) :
		up = torch.zeros( (x.size( 0 ) , x.size( 1 ) , x.size( 2 ) * 2 , x.size( 3 ) * 2) , device = x.device )
		up[ : , : , : :2 , : :2 ] = x * 4
		
		return self.conv_gauss( up , self.kernel )
	
	def pyramid_decom( self , img ) :
		self.kernel = self.kernel.to( img.device )
		current = img
		down = self.pyramid_down( current )
		diff = current - self.upsample( down )
		current = down
		
		return diff,current # HF LF
	
	def pyramid_recons( self , HF,LF ) :
		image = LF
		up = self.upsample( image )
		
		return up + HF
		

class DSFD( nn.Module ) :
	"""Single Shot Multibox Architecture
	The network is composed of a base VGG network followed by the
	added multibox conv layers.  Each multibox layer branches into
		1) conv2d for class conf scores
		2) conv2d for localization predictions
		3) associated priorbox layer to produce default bounding
		   boxes specific to the layer's feature map size.
	See: https://arxiv.org/pdf/1512.02325.pdf for more details.

	Args:
		phase: (string) Can be "test" or "train"
		size: input image size
		base: VGG16 layers for input, size of either 300 or 500
		extras: extra layers that feed to multibox loc and conf layers
		head: "multibox head" consists of loc and conf conv layers
	"""
	
	def __init__( self , phase , base , extras , fem , head1 , head2 , num_classes ) :
		super( DSFD , self ).__init__()
		self.phase = phase
		self.num_classes = num_classes
		self.vgg = nn.ModuleList( base )
		
		self.L2Normof1 = L2Norm( 256 , 10 )
		self.L2Normof2 = L2Norm( 512 , 8 )
		self.L2Normof3 = L2Norm( 512 , 5 )
		
		self.extras = nn.ModuleList( extras )
		self.fpn_topdown = nn.ModuleList( fem[ 0 ] )
		self.fpn_latlayer = nn.ModuleList( fem[ 1 ] )
		
		self.fpn_fem = nn.ModuleList( fem[ 2 ] )
		
		self.L2Normef1 = L2Norm( 256 , 10 )
		self.L2Normef2 = L2Norm( 512 , 8 )
		self.L2Normef3 = L2Norm( 512 , 5 )
		
		self.loc_pal1 = nn.ModuleList( head1[ 0 ] )  # nn.ModuleList是一种存储子模块的工具
		self.conf_pal1 = nn.ModuleList( head1[ 1 ] )
		
		self.loc_pal2 = nn.ModuleList( head2[ 0 ] )
		self.conf_pal2 = nn.ModuleList( head2[ 1 ] )
		
		# 输入64通道，尺度缩减x2
		self.HF = nn.Sequential(
				nn.Conv2d( 64 , 64 , kernel_size = 3 , padding = 1 ) ,
				nn.ReLU( inplace = True ) ,
				Interpolate( 2 ) ,  # 上采样
				nn.Conv2d( 64 , 3 , kernel_size = 3 , padding = 1 ) ,
				nn.Sigmoid()
		)
		self.LF = nn.Sequential(
				nn.Conv2d( 64 , 64 , kernel_size = 3 , padding = 1 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( 64 , 3 , kernel_size = 3 , padding = 1 ) ,
				nn.Sigmoid()
		)
		
		self.Lap = Lap_Pyramid_Conv()
		
		# 计算teacher模型和学生模型的KL散度
		self.KL = DistillKL( T = 4.0 )
		
		if self.phase == 'test' :
			self.softmax = nn.Softmax( dim = -1 )
			self.detect = Detect( cfg )
	
	def _upsample_prod( self , x , y ) :
		_ , _ , H , W = y.size()
		return F.upsample( x , size = (H , W) , mode = 'bilinear' ) * y
	
	# 反射图解码通路
	def enh_forward( self , x ) :
		
		x = x[ :1 ]
		for k in range( 5 ) :
			x = self.vgg[ k ]( x )
		
		R = self.ref( x )
		
		return R
	
	def test_forward( self , x ) :
		size = x.size()[ 2 : ]
		pal1_sources = list()
		pal2_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()
		
		for k in range( 16 ) :
			x = self.vgg[ k ]( x )
			# x检测通路的输入
			if k == 4 :
				x_dark = x  # xlight、xdark分解通路的输入
		
		HF_dark_decoder = self.HF( x_dark )
		
		# print( '暗图' )
		# image = np.transpose( R_dark[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )
		# # 保存图像到文件
		# plt.savefig( f'train_暗图.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		#
		# print( '亮图' )
		# image = np.transpose( R_light[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )
		# # 保存图像到文件
		# plt.savefig( f'train_亮图.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		
		# the following is the rest of the original detection pipeline
		of1 = x
		s = self.L2Normof1( of1 )
		pal1_sources.append( s )
		# apply vgg up to fc7
		for k in range( 16 , 23 ) :
			x = self.vgg[ k ]( x )
		of2 = x
		s = self.L2Normof2( of2 )
		pal1_sources.append( s )
		
		for k in range( 23 , 30 ) :
			x = self.vgg[ k ]( x )
		of3 = x
		s = self.L2Normof3( of3 )
		pal1_sources.append( s )
		
		for k in range( 30 , len( self.vgg ) ) :
			x = self.vgg[ k ]( x )
		of4 = x
		pal1_sources.append( of4 )
		# apply extra layers and cache source layer outputs
		
		for k in range( 2 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of5 = x
		pal1_sources.append( of5 )
		for k in range( 2 , 4 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of6 = x
		pal1_sources.append( of6 )
		
		conv7 = F.relu( self.fpn_topdown[ 0 ]( of6 ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 1 ]( conv7 ) , inplace = True )
		conv6 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 0 ]( of5 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 2 ]( conv6 ) , inplace = True )
		convfc7_2 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 1 ]( of4 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 3 ]( convfc7_2 ) , inplace = True )
		conv5 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 2 ]( of3 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 4 ]( conv5 ) , inplace = True )
		conv4 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 3 ]( of2 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 5 ]( conv4 ) , inplace = True )
		conv3 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 4 ]( of1 ) ) , inplace = True )
		
		ef1 = self.fpn_fem[ 0 ]( conv3 )
		ef1 = self.L2Normef1( ef1 )
		ef2 = self.fpn_fem[ 1 ]( conv4 )
		ef2 = self.L2Normef2( ef2 )
		ef3 = self.fpn_fem[ 2 ]( conv5 )
		ef3 = self.L2Normef3( ef3 )
		ef4 = self.fpn_fem[ 3 ]( convfc7_2 )
		ef5 = self.fpn_fem[ 4 ]( conv6 )
		ef6 = self.fpn_fem[ 5 ]( conv7 )
		
		pal2_sources = (ef1 , ef2 , ef3 , ef4 , ef5 , ef6)
		for (x , l , c) in zip( pal1_sources , self.loc_pal1 , self.conf_pal1 ) :
			loc_pal1.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal1.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		for (x , l , c) in zip( pal2_sources , self.loc_pal2 , self.conf_pal2 ) :
			loc_pal2.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal2.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		features_maps = [ ]
		for i in range( len( loc_pal1 ) ) :
			feat = [ ]
			feat += [ loc_pal1[ i ].size( 1 ) , loc_pal1[ i ].size( 2 ) ]
			features_maps += [ feat ]
		
		loc_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal1 ] , 1 )
		conf_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal1 ] , 1 )
		
		loc_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal2 ] , 1 )
		conf_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal2 ] , 1 )
		
		priorbox = PriorBox( size , features_maps , cfg , pal = 1 )
		self.priors_pal1 = priorbox.forward().requires_grad_( False )
		
		priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
		self.priors_pal2 = priorbox.forward().requires_grad_( False )
		
		if self.phase == 'test' :
			output = self.detect.forward( loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) , self.softmax( conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ) ,  # conf preds
			                              self.priors_pal2.type( type( x.data ) ) )
		
		else :
			output = (loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) , conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) , self.priors_pal1 , loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
			          conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) , self.priors_pal2)
		
		# packing the outputs from the reflectance decoder:
		return output , HF_dark_decoder
	
	# during training, the model takes the paired images, and their pseudo GT illumination maps from the Retinex Decom Net
	def forward( self , x , x_light ) : # LF_dark_Lap , LF_light_Lap
		size = x.size()[ 2 : ]
		pal1_sources = list()
		pal2_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()
		
		HF_dark_Lap,LF_dark_Lap = self.Lap.pyramid_decom( x)
		HF_light_Lap,LF_light_Lap = self.Lap.pyramid_decom( x_light)
		
		# apply vgg up to conv4_3 relu
		# x输入暗图 xlight输入亮图
		for k in range( 5 ) :
			x_light = self.vgg[ k ]( x_light )
		
		for k in range( 16 ) :
			x = self.vgg[ k ]( x )
			# x检测通路的输入
			if k == 4 :
				x_dark = x  # xlight、xdark分解通路的输入
		
		HF_dark_decoder = self.HF( x_dark )
		HF_light_decoder = self.HF( x_light )
		
		
		# Interchange
		# I是Retinex Net生成的低光照下的光照图
		x_dark_recon = self.Lap.pyramid_recons(HF_light_decoder , LF_dark_Lap).detach()
		x_light_recon = self.Lap.pyramid_recons(HF_dark_decoder , LF_light_Lap).detach()
		
		for k in range( 5 ) :
			x_light_recon = self.vgg[ k ]( x_light_recon )
		for k in range( 5 ) :
			x_dark_recon = self.vgg[ k ]( x_dark_recon )
		
		# Redecomposition
		# 重新分解
		HF_light_recon = self.HF( x_light_recon )
		HF_dark_recon = self.HF( x_dark_recon )
		
		# mutual feature alignment loss
		x_light = x_light.flatten( start_dim = 2 ).mean( dim = -1 )
		x_dark = x_dark.flatten( start_dim = 2 ).mean( dim = -1 )
		x_light_recon = x_light_recon.flatten( start_dim = 2 ).mean( dim = -1 )
		x_dark_recon = x_dark_recon.flatten( start_dim = 2 ).mean( dim = -1 )
		# 经过网络提取特征后的KL散度损失
		loss_mutual = cfg.WEIGHT.MC * (self.KL( x_light , x_dark ) + self.KL( x_dark , x_light ) + self.KL( x_light_recon , x_dark_recon ) + self.KL( x_dark_recon , x_light_recon ))
		
		# the following is the rest of the original detection pipeline
		of1 = x
		s = self.L2Normof1( of1 )
		pal1_sources.append( s )
		# apply vgg up to fc7
		for k in range( 16 , 23 ) :
			x = self.vgg[ k ]( x )
		of2 = x
		s = self.L2Normof2( of2 )
		pal1_sources.append( s )
		
		for k in range( 23 , 30 ) :
			x = self.vgg[ k ]( x )
		of3 = x
		s = self.L2Normof3( of3 )
		pal1_sources.append( s )
		
		for k in range( 30 , len( self.vgg ) ) :
			x = self.vgg[ k ]( x )
		of4 = x
		pal1_sources.append( of4 )
		# apply extra layers and cache source layer outputs
		
		for k in range( 2 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of5 = x
		pal1_sources.append( of5 )
		for k in range( 2 , 4 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of6 = x
		pal1_sources.append( of6 )
		
		conv7 = F.relu( self.fpn_topdown[ 0 ]( of6 ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 1 ]( conv7 ) , inplace = True )
		conv6 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 0 ]( of5 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 2 ]( conv6 ) , inplace = True )
		convfc7_2 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 1 ]( of4 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 3 ]( convfc7_2 ) , inplace = True )
		conv5 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 2 ]( of3 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 4 ]( conv5 ) , inplace = True )
		conv4 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 3 ]( of2 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 5 ]( conv4 ) , inplace = True )
		conv3 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 4 ]( of1 ) ) , inplace = True )
		
		ef1 = self.fpn_fem[ 0 ]( conv3 )
		ef1 = self.L2Normef1( ef1 )
		ef2 = self.fpn_fem[ 1 ]( conv4 )
		ef2 = self.L2Normef2( ef2 )
		ef3 = self.fpn_fem[ 2 ]( conv5 )
		ef3 = self.L2Normef3( ef3 )
		ef4 = self.fpn_fem[ 3 ]( convfc7_2 )
		ef5 = self.fpn_fem[ 4 ]( conv6 )
		ef6 = self.fpn_fem[ 5 ]( conv7 )
		
		pal2_sources = (ef1 , ef2 , ef3 , ef4 , ef5 , ef6)
		for (x , l , c) in zip( pal1_sources , self.loc_pal1 , self.conf_pal1 ) :
			loc_pal1.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal1.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		for (x , l , c) in zip( pal2_sources , self.loc_pal2 , self.conf_pal2 ) :
			loc_pal2.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal2.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		features_maps = [ ]
		for i in range( len( loc_pal1 ) ) :
			feat = [ ]
			feat += [ loc_pal1[ i ].size( 1 ) , loc_pal1[ i ].size( 2 ) ]
			features_maps += [ feat ]
		
		loc_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal1 ] , 1 )
		conf_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal1 ] , 1 )
		
		loc_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal2 ] , 1 )
		conf_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal2 ] , 1 )
		
		priorbox = PriorBox( size , features_maps , cfg , pal = 1 )
		self.priors_pal1 = priorbox.forward().requires_grad_( False )
		
		priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
		self.priors_pal2 = priorbox.forward().requires_grad_( False )
		
		if self.phase == 'test' :
			output = self.detect.forward( loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) , self.softmax( conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ) ,  # conf preds
					self.priors_pal2.type( type( x.data ) ) )
		
		else :
			output = (loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) , conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) , self.priors_pal1 , loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
			          conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) , self.priors_pal2)
		
		recon_dark = self.Lap.pyramid_recons( HF_dark_decoder , LF_dark_Lap )
		recon_light = self.Lap.pyramid_recons( HF_light_decoder , LF_light_Lap )
		
		# packing the outputs from the reflectance decoder:
		return (output ,
		        [ HF_dark_decoder , HF_light_decoder , HF_dark_recon , HF_light_recon ] ,
		        [HF_dark_Lap,LF_dark_Lap,HF_light_Lap,LF_light_Lap], loss_mutual,
		        [ recon_dark , recon_light]
		        )
	
	def load_weights( self , base_file ) :
		other , ext = os.path.splitext( base_file )
		if ext == '.pkl' or '.pth' :
			print( 'Loading weights into state dict...' )
			mdata = torch.load( base_file , map_location = lambda storage , loc : storage )
			
			epoch = 0
			self.load_state_dict( mdata )
			print( 'Finished!' )
		else :
			print( 'Sorry only .pth and .pkl files supported.' )
		return epoch
	
	def xavier( self , param ) :
		init.xavier_uniform_( param )
		
	def kaiming( self,param  ):
		init.kaiming_uniform_( param )
	
	def weights_init( self , m ,mothod='xavier') :
		if isinstance( m , nn.Conv2d ) :
			if mothod == 'kaiming':
				self.kaiming( m.weight.data)
			elif mothod == 'xavier':
				self.xavier( m.weight.data )
			m.bias.data.zero_()
		
		if isinstance( m , nn.ConvTranspose2d ) :
			if mothod == 'kaiming':
				self.kaiming( m.weight.data)
			elif mothod == 'xavier':
				self.xavier( m.weight.data )
				
			if 'bias' in m.state_dict().keys() :
				m.bias.data.zero_()
		
		if isinstance( m , nn.BatchNorm2d ) :
			m.weight.data[ ... ] = 1
			m.bias.data.zero_()


vgg_cfg = [ 64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 256 , 'C' , 512 , 512 , 512 , 'M' , 512 , 512 , 512 , 'M' ]

extras_cfg = [ 256 , 'S' , 512 , 128 , 'S' , 256 ]

fem_cfg = [ 256 , 512 , 512 , 1024 , 512 , 256 ]


def fem_module( cfg ) :
	topdown_layers = [ ]
	lat_layers = [ ]
	fem_layers = [ ]
	
	topdown_layers += [ nn.Conv2d( cfg[ -1 ] , cfg[ -1 ] , kernel_size = 1 , stride = 1 , padding = 0 ) ]
	for k , v in enumerate( cfg ) :
		fem_layers += [ FEM( v ) ]
		cur_channel = cfg[ len( cfg ) - 1 - k ]
		if len( cfg ) - 1 - k > 0 :
			last_channel = cfg[ len( cfg ) - 2 - k ]
			topdown_layers += [ nn.Conv2d( cur_channel , last_channel , kernel_size = 1 , stride = 1 , padding = 0 ) ]
			lat_layers += [ nn.Conv2d( last_channel , last_channel , kernel_size = 1 , stride = 1 , padding = 0 ) ]
	return (topdown_layers , lat_layers , fem_layers)


def vgg( cfg , i , batch_norm = False ) : # 修改后，vgg的索引应该修改
	layers = [ ]
	in_channels = i
	for v in cfg :
		if v == 'M' :
			layers += [ nn.MaxPool2d( kernel_size = 2 , stride = 2 ) ]
		elif v == 'C' :
			layers += [ nn.MaxPool2d( kernel_size = 2 , stride = 2 , ceil_mode = True ) ]
		else :
			conv2d = nn.Conv2d( in_channels , v , kernel_size = 3 , padding = 1 )
			if batch_norm :
				layers += [ conv2d , nn.BatchNorm2d( v ) , nn.ReLU( inplace = True ) ]
			else :
				layers += [ conv2d , nn.ReLU( inplace = True ) ]
			in_channels = v
	conv6 = nn.Conv2d( 512 , 1024 , kernel_size = 3 , padding = 3 , dilation = 3 )
	conv7 = nn.Conv2d( 1024 , 1024 , kernel_size = 1 )
	layers += [ conv6 , nn.ReLU( inplace = True ) , conv7 , nn.ReLU( inplace = True ) ]
	return layers


def add_extras( cfg , i , batch_norm = False ) :
	# Extra layers added to VGG for feature scaling
	layers = [ ]
	in_channels = i
	flag = False
	for k , v in enumerate( cfg ) :
		if in_channels != 'S' :
			if v == 'S' :
				layers += [ nn.Conv2d( in_channels , cfg[ k + 1 ] , kernel_size = (1 , 3)[ flag ] , stride = 2 , padding = 1 ) ]
			else :
				layers += [ nn.Conv2d( in_channels , v , kernel_size = (1 , 3)[ flag ] ) ]
			flag = not flag
		in_channels = v
	return layers


def multibox( vgg , extra_layers , num_classes ) :
	loc_layers = [ ]
	conf_layers = [ ]
	vgg_source = [ 14 , 21 , 28 , -2 ]
	
	for k , v in enumerate( vgg_source ) :
		loc_layers += [ nn.Conv2d( vgg[ v ].out_channels , 4 , kernel_size = 3 , padding = 1 ) ]
		conf_layers += [ nn.Conv2d( vgg[ v ].out_channels , num_classes , kernel_size = 3 , padding = 1 ) ]
	for k , v in enumerate( extra_layers[ 1 : :2 ] , 2 ) :
		loc_layers += [ nn.Conv2d( v.out_channels , 4 , kernel_size = 3 , padding = 1 ) ]
		conf_layers += [ nn.Conv2d( v.out_channels , num_classes , kernel_size = 3 , padding = 1 ) ]
	return (loc_layers , conf_layers)


def build_net_dark( phase , num_classes = 2 ) :
	base = vgg( vgg_cfg , 3 )
	extras = add_extras( extras_cfg , 1024 )
	head1 = multibox( base , extras , num_classes )
	head2 = multibox( base , extras , num_classes )
	fem = fem_module( fem_cfg )
	return DSFD( phase , base , extras , fem , head1 , head2 , num_classes )


class DistillKL( nn.Module ) :
	"""KL divergence for distillation"""
	
	# 知识蒸馏模块，处理KL散度
	def __init__( self , T ) :
		super( DistillKL , self ).__init__()
		self.T = T
	
	def forward( self , y_s , y_t ) :
		# y_s学生模型的输出，y_t 教师模型的输出
		p_s = F.log_softmax( y_s / self.T , dim = 1 )  # 对数概率分布
		p_t = F.softmax( y_t / self.T , dim = 1 )  # 概率分布
		# 计算KL散度
		# size_average不使用平均损失，而是返回总损失，(self.T ** 2)补偿温度缩放，/ y_s.shape[0]计算平均损失
		loss = F.kl_div( p_s , p_t , size_average = False ) * (self.T ** 2) / y_s.shape[ 0 ]
		return loss

# 在数据加载后添加检查
def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
