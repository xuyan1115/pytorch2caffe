# -*- coding: utf-8
import sys
import numpy as np
import os

import torch
from torch.autograd import Variable

from pytorch2caffe import plot_graph, pytorch2caffe
from model.PCBModel import PCBModel as Model

import caffe

# test the model or generate model
test_mod = True

caffemodel_dir = 'demo'
input_size = (1, 3, 384, 128)

model_def = './demo/bpm_reid-pytorch2caffe.prototxt'
model_weights = './demo/bpm_reid-pytorch2caffe.caffemodel'
input_name = 'ConvNdBackward1'
output_name = ['BatchNormBackward192', 'BatchNormBackward198', 'BatchNormBackward204', 'BatchNormBackward210',
				'BatchNormBackward216', 'BatchNormBackward222']
# pytorch net
model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256, num_classes=751)
model.load_state_dict(torch.load('../beyond-part-models/ckpt.pth')['state_dicts'][0])
model.eval()

# random input
image = np.random.randint(0, 255, input_size) / 255.
input_data = image.astype(np.float32)

# pytorch forward
input_var = Variable(torch.from_numpy(input_data))

if not test_mod:
    # generate caffe model
    output_var = model(input_var)
    plot_graph(output_var, os.path.join(caffemodel_dir, 'pytorch_graph.dot'))
    pytorch2caffe(input_var, output_var, model_def, model_weights)
    exit(0)

# test caffemodel
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

net.blobs['data'].data[...] = input_data

net.forward(start=input_name)
caffe_output = [net.blobs[name].data for name in output_name]
caffe_output = np.array(caffe_output)[:, :, :, 0, 0]

model = model.cuda()
input_var = input_var.cuda()
output_var, _ = model(input_var)
pytorch_output = [var.data.cpu().numpy() for var in output_var]
pytorch_output = np.array(pytorch_output)

print(input_size, pytorch_output.shape, caffe_output.shape)
print('pytorch: min: {}, max: {}, mean: {}'.format(pytorch_output.min(), pytorch_output.max(), pytorch_output.mean()))
print('  caffe: min: {}, max: {}, mean: {}'.format(caffe_output.min(), caffe_output.max(), caffe_output.mean()))

diff = np.abs(pytorch_output - caffe_output)
print('   diff: min: {}, max: {}, mean: {}, median: {}'.format(diff.min(), diff.max(), diff.mean(), np.median(diff)))
