import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torchsummary
import random


## preparing arguments
import sys
import getopt
arg_list=sys.argv[1:]
norm_dict={"bn":batchNorm2d, "in":instanceNorm2d, "bin":batchInstanceNormalization2d, "ln":layerNorm2d, "gn":groupNorm2d, "nn":NoNorm, "inbuilt":nn.BatchNorm2d}

short_options=""
long_options=["normalization=", "model_file=", "test_data_file=", "output_file=", "n="]

try:
    input_args, values=getopt.getopt(arg_list, short_options, long_options)
except getopt.error as err:
    print("arguments error")

arg_dict={}
for x, y in input_args:
    if x=='--n':
        arg_dict['n']=y
    elif x=="--normalization":
        arg_dict["Normalizer"]=norm_dict[x]
    elif x=="--output_file":
        arg_dict["output_file"]=y
    elif x=="--test_data_file":
        arg_dict["test_data_file"]=y
    elif x=="--model_file":
        arg_dict["model_file"]=y




# preparing data

np_data=np.genfromtxt(arg_dict["test_data_file"], delimiter=',').astype(np.float32)
new_dat=np.reshape(np_data, (len(np_data), 3, 32, 32))

output=np.asarray([])


from modelv4 import batchNorm2d, instanceNorm2d, layerNorm2d, batchInstanceNormalization2d, groupNorm2d, NoNorm, JugaadResNet

bn_props={'affine':True, 'track_running_stats':True, 'eps':1e-5, 'momentum':0.1}
in_props={'affine':False, 'track_running_stats':False, 'eps':1e-5, 'momentum':0.1}
bin_props={'affine':True, 'track_running_stats':False, 'eps':1e-5, 'momentum':0.1}
ln_props={'elementwise_affine':True, 'eps':1e-5, 'momentum':0.1}
gn_props={'num_groups':8, 'eps':1e-5, 'affine':True}

normProps={}
if arg_dict["Normalizer"]==nn.BatchNorm2d or arg_dict["Normalizer"]==batchNorm2d:
    normProps=bn_props
elif arg_dict["Normalizer"]==layerNorm2d:
    normProps=ln_props
elif arg_dict["Normalizer"]==groupNorm2d:
    normProps=gn_props
elif arg_dict["Normalizer"]==instanceNorm2d:
    normProps=in_props
elif arg_dict["Normalizer"]=batchInstanceNormalization2d:
    normProps=bin_props

net=JugaadResNet(arg_dict["n"], 10, arg_dict["Normalizer"], normProps).cuda()
net.load_state_dict(torch.load(arg_dict["model_file"]))
net.eval()


for i in range(0, len(new_dat), 128):
    st_index=i
    end_index=min(i+128, len(new_dat))
    inpt=torch.from_numpy(new_dat[st_index:end_index]).cuda()
    out=net(inpt)
    _, out1=torch.max(out, 1)
    to_add_out=out1.cpu().numpy()
    res=np.append(res, to_add_out)



np.savetxt(arg_dict["output_file"], res.astype(np.int32))