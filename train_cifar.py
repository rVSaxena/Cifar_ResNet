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
short_options=""
long_options=["normalization=", "data_dir=", "output_file=", "n="]
try:
    input_args, values=getopt.getopt(arg_list, short_options, long_options)
except getopt.error as err:
    print("arguments error")
arg_dict={}
norm_dict={"bn":batchNorm2d, "in":instanceNorm2d, "bin":batchInstanceNormalization2d, "ln":layerNorm2d, "gn":groupNorm2d, "nn":NoNorm, "torch_nn":nn.BatchNorm2d}
for x, y in input_args:
    if x=='--n':
        arg_dict['n']=y
    elif x=="--normalization":
        arg_dict["Normalizer"]=norm_dict[x]
    elif x=="--output_file":
        arg_dict["output_file"]=y
    elif x=="--data_dir":
        arg_dict["data_dir"]=y


# input_args has all the required stuff

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
pp_mean=torch.from_numpy(np.load("pp_mean.npy").astype(np.float32))
def subtract_pp_mean(x):
    return x-pp_mean

transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(subtract_pp_mean)])
t_set=torchvision.datasets.CIFAR10(root=arg_dict["data_dir"], train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(t_set, [40000,10000],
                                          generator=torch.Generator().manual_seed(42))
# choosing 42 as the seed is important, as 42 is the answer to everything.
trainloader=torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)
valloader=torch.utils.data.DataLoader(valset, batch_size=128,
                                          shuffle=True)

def save_nn(nonExtensionPath, model, n=None):
    """
    nonExtensionPath: str, the path except any extension, eg .pth
    model:
    """
    if n is None:
        n=model.n
    with open(nonExtensionPath+'.txt', 'w+') as f:
        f.write(str(n)+'\n')
        f.write(str(model.num_classes))
    torch.save(model.state_dict(), nonExtensionPath+'.pth')
    return

def load_nn(nonExtensionPath, modelClass, to_cuda=False):
    with open(nonExtensionPath+'.txt', 'r') as f:
        n=int(f.readline())
        r=int(f.readline())
    if to_cuda:
        model=modelClass(n, r).cuda()
    else:
        model=modelClass(n, r)
    model.load_state_dict(torch.load(nonExtensionPath+'.pth'))
    return model
    
    




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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

if arg_dict["Normalizer"]==batchInstanceNormalization2d:
    bin_gates = [p for p in net.parameters() if getattr(p, 'bin_gate', False)]

net.train()
for epoch in range(100):  # loop over the dataset multiple times

    custom_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(hd_acc), data[1].to(hd_acc)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if arg_dict["Normalizer"]==batchInstanceNormalization2d:
            for p in bin_gates:
                p.data.clamp_(min=0, max=1)
        
        custom_loss=(i*custom_loss+loss.item())/(i+1)

net.eval()
torch.save(net.state_dict(), arg_dict["output_file"])

