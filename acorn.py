###ACORN implementation from Stanford's Computational Imaging Lab
###http://www.computationalimaging.org/publications/acorn/
###also gotta credit lucidrains for a lot of this code ("if it ain't broke, don't fix it")

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def exists(val):
	return val is not None

#Activation of layers for ACORN - default is Sine
class LayerActivation(nn.Module):
	def __init__(self, torch_activation=torch.sin, w0=30):
		super().__init__()
		self.activation = torch_activation
		self.w0 = w0

	def forward(self, input):
		return self.activation(self.w0 * input)


#ACORN layer

class AcornLayer(nn.Module):
	def __init__(self, in_features, out_features, w0 = 1., c = 6., use_bias=True, is_first=False, activation = None):
		super().__init__()
		self.in_features = in_features
		self.is_first = is_first

		weight = torch.zeros(out_features, in_features)
		bias = torch.zeros(out_features) if use_bias else None
		self.init_(weight, bias, c = c, w0 = w0)

		self.weight = nn.Parameter(weight)
		self.bias = nn.Parameter(bias) if use_bias else None
		self.activation = LayerActivation(w0=w0) if activation is None else activation

	def init_(self, weight, bias, c, w0):
		feat = self.in_features

		w_std = (1 / feat) if self.is_first else (math.sqrt(c / feat) / w0)
		weight.uniform_(-w_std, w_std)

		if exists(bias):
			bias.uniform_(-w_std, w_std)

	def forward(self, x):
		out = F.linear(x, self.weight, self.bias)
		out = self.activation(out)
		return out

#ACORN network

class AcornNetwork(nn.Module):
	def __init__(self, in_features, hidden_features, out_features, num_layers, w0=1., w0_initial=30., use_bias=True, final_activation=None):
		super().__init__()
		self.num_layers = num_layers
		self.hidden_features = hidden_features

		self.layers = nn.ModuleList([])
		for layer in range(num_layers):
			is_first = layer == 0
			layer_w0 = w0_initial if is_first else w0
			layer_features = in_features if is_first else hidden_features

			self.layers.append(AcornLayer(in_features = layer_features,
				out_features = hidden_features,
				w0 = layer_w0,
				use_bias = use_bias,
				is_first = is_first))

		final_activation = nn.Identity() if not exists(final_activation) else final_activation
		self.final_layer = AcornLayer(in_features = hidden_features,
			out_features = out_features,
			w0 = w0,
			use_bias = use_bias,
			activation = final_activation)

	def forward(self, x):
		#no modulation here
		x = layer(x)
		return self.last_layer(x)


#ACORN wrapper. Things get actually different here

class AcornWrapper(nn.Module):
	def __init__(self, net1, net2, image_width, image_height):
		super().__init__()
		assert isinstance(net, AcornNetwork), "AcornWrapper must recieve an ACORN network"

		self.net1 = net1
		self.net2 = net2
		self.image_width = image_width
		self.image_height = image_height

		#create mapping grid
		tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
		mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
		mgrid = rearrange(mgrid, 'h w c -> (h w) c')
		self.register_buffer('grid', mgrid)
		
	def forward(self, img = None, *):
		coords = self.grid.clone().detach().requires_grad_()
		fine_coords
		out = self.net(coords)
		out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

		if exists(img):
			return F.mse_loss(img, out)

		return out


