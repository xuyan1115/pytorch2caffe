import torch
from torch.autograd import Variable

import os
from model.PCBModel import PCBModel as Model
from pytorch2caffe import pytorch2caffe, plot_graph


model = Model(last_conv_stride=1, num_stripes=6, local_conv_out_channels=256, num_classes=751)
model.load_state_dict(torch.load('../beyond-part-models/ckpt.pth')['state_dicts'][0])
model.eval()

input_var = Variable(torch.randn(1, 3, 384, 128))
output_var, _ = model(input_var)
print(type(output_var))

output_dir = 'demo'
#plot_graph(output_var, os.path.join(output_dir, 'bpm_reid.dot'))

pytorch2caffe(input_var, output_var, 
        os.path.join(output_dir, 'bpm_reid-pytorch2caffe.prototxt'), 
        os.path.join(output_dir, 'bpm_reid-pytorch2caffe.caffemodel'))
