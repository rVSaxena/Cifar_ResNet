

"""
Dimensions as: (N, C, H, W)
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter as param
import torch.nn.functional as F


class instanceNorm2d(nn.Module):

    def __init__(self, num_features=None, affine=False, track_running_stats=False, eps=1e-5, momentum=0.1):
        """
        momentum=None not implemented
        """
        super(instanceNorm2d, self).__init__()
        self.register_buffer('eps', torch.tensor(eps, requires_grad=False))
        self.register_buffer('momentum', torch.tensor(momentum, requires_grad=False))
        self.affine=affine
        self.track_running_stats=track_running_stats

        if self.affine:
            self.gamma=param(torch.ones(num_features, dtype=torch.float32, requires_grad=True)+torch.rand(num_features)/4)
            self.beta=param(torch.rand(num_features, requires_grad=True)/4)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, requires_grad=False))
            self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float32, requires_grad=False))

        return

    def forward(self, x):

        # first update running variables
        # always expectd 4D input in the form-NxCxHxW

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*torch.mean(x, dim=(0, 2, 3))
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*torch.var(x, dim=(0,2,3), unbiased=False)

        if self.training:
            x=(x-x.mean(dim=(2, 3), keepdim=True))/torch.sqrt(x.var(dim=(2,3), keepdim=True, unbiased=False)+self.eps)
            if self.affine:
                x=x*self.gamma[None,:,None,None]+self.beta[None,:,None,None]

        else:
            with torch.no_grad():
                if self.track_running_stats:
                    x=(x-self.running_mean[None,:,None,None])/torch.sqrt(self.eps+self.running_var[None,:,None,None])
                else:
                    x=(x-x.mean(dim=(2, 3), keepdim=True))/torch.sqrt(x.var(dim=(2,3), keepdim=True, unbiased=False)+self.eps)

                if self.affine:
                     x=x*self.gamma[None,:,None,None]+self.beta[None,:,None,None]

        return x


class batchNorm2d(nn.Module):

    def __init__(self, num_features=None, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1):
        super(batchNorm2d, self).__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))
        self.register_buffer("momentum", torch.tensor(momentum, dtype=torch.float32))
        self.affine=affine
        self.track_running_stats=track_running_stats
        

        if self.affine:
            self.register_parameter('gamma', param(torch.ones(num_features, dtype=torch.float32, requires_grad=True)))
            self.register_parameter('beta', param(torch.zeros(num_features, requires_grad=True)))

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float32))

        return

    def forward(self, x):
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*torch.mean(x, dim=(0,2,3))
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*torch.var(x, dim=(0,2,3), unbiased=False)

        if self.training:
            x=(x-x.mean(dim=(0,2,3), keepdim=True))/torch.sqrt(self.eps+x.var(dim=(0,2,3), keepdim=True, unbiased=False))
            if self.affine:
                x=x*self.gamma[None,:,None,None]+self.beta[None,:,None,None]
        else:
            with torch.no_grad():
                if self.track_running_stats:
                    x=(x-self.running_mean[None,:,None,None])/torch.sqrt(self.eps+self.running_var[None,:,None,None])
                else:
                    x=(x-x.mean(dim=(0,2,3), keepdim=True))/torch.sqrt(self.eps+x.var(dim=(0,2,3), keepdim=True, unbiased=False))

                if self.affine:
                    x=x*self.gamma[None,:,None,None]+self.beta[None,:,None,None]

        return x


class layerNorm2d(nn.Module):

    def __init__(self, elementwise_affine=True, eps=1e-5, momentum=0.1):
        super(layerNorm2d, self).__init__()
        self.register_buffer('eps', torch.tensor(eps, requires_grad=False))
        self.register_buffer('momentum', torch.tensor(momentum, requires_grad=False))
        self.elementwise_affine=True

        if self.elementwise_affine:
            self.gamma=param(torch.ones(1, dtype=torch.float32, requires_grad=True)+torch.rand(1)/4)
            self.beta=param(torch.rand(1, requires_grad=True)/4)

        return

    def forward(self, x):
        if self.training:
            x=(x-x.mean(dim=(1,2,3), keepdim=True))/torch.sqrt(self.eps+x.var(dim=(1,2,3), keepdim=True, unbiased=False))
            x=x*self.gamma+self.beta
        else:
            with torch.no_grad():
                x=(x-x.mean(dim=(1,2,3), keepdim=True))/torch.sqrt(self.eps+x.var(dim=(1,2,3), keepdim=True, unbiased=False))
                x=x*self.gamma+self.beta
        return x


class batchInstanceNormalization2d(nn.Module):

    def __init__(self, num_features=None, affine=True, track_running_stats=False, eps=1e-5, momentum=0.1):
        super(batchInstanceNormalization2d, self).__init__()

        self.affine=affine
        self.BN=batchNorm2d(num_features, False, track_running_stats, eps, momentum)
        self.IN=instanceNorm2d(num_features, False, track_running_stats, eps, momentum)

        self.gate=param(torch.rand(1, requires_grad=True))
        setattr(self.gate, 'bin_gate', True)

        if self.affine:
            self.gamma=param(torch.ones(num_features, dtype=torch.float32, requires_grad=True)+torch.rand(num_features)/4)
            self.beta=param(torch.rand(num_features, requires_grad=True)/4)

        return

    def forward(self, x):
        
        if self.training:
            a=self.BN(x)
            b=self.IN(x)
            x=self.gate*a+(1-self.gate)*b
        else:
            with torch.no_grad():
                a=self.BN(x)
                b=self.IN(x)
                x=self.gate*a+(1-self.gate)*b

        return x


class groupNorm2d(nn.Module):

    def __init__(self, num_groups=None, num_features=None, h=None, w=None, eps=1e-5, affine=True):
        """
        only params, no buffers
        """
        super(groupNorm2d, self).__init__()
        self.register_buffer('num_groups', torch.tensor(num_groups, requires_grad=False))
        self.register_buffer('eps', torch.tensor(eps, requires_grad=False))
        self.register_buffer('num_features', torch.tensor(num_features, requires_grad=False))
        self.affine=affine
        
        if self.affine:
            self.gamma=param(torch.ones(num_features, dtype=torch.float32, requires_grad=True)+torch.rand(num_features)/4)
            self.beta=param(torch.rand(num_features, requires_grad=True)/4)

        return

    def forward(self, x):
        n, c, h, w=x.shape

        x=torch.reshape(x, (n, self.num_groups, c//self.num_groups, h, w))
        x=(x-x.mean(dim=(2, 3, 4), keepdim=True))/torch.sqrt(self.eps+x.var(dim=(2,3,4), keepdim=True))

        x=torch.reshape(x, (n, c, h, w))
        return x

class NoNorm(nn.Module):

    def __init__(self, **kwargs):
        super(NoNorm, self).__init__()
        return

    def forward(self, x):
        return x

class JugaadResNet(nn.Module):

    @staticmethod
    def get_padding_size():
        """
        Return the padding to be done to get out_dim
        from in_dim given kernel_dim and stride.
        """
        padding={}
        padding[(32,32,3,1)]=1
        padding[(32,16,3,2)]=1
        padding[(16,16,3,1)]=1
        padding[(16,8,3,2)]=1
        padding[(8,8,3,1)]=1

        return padding


    def __init__(self, n, r, Normalizer=NoNorm, NormalizerProps={}):
        super(JugaadResNet, self).__init__()
        self.n=n
        self.num_classes=r
        self.padding_info=self.get_padding_size()
        
        # SAK stands for size_adapting_kernel
        self.SAK1=nn.Conv2d(16,32,1,stride=2, bias=False)
        # self.SAK1.weight=nn.Parameter(torch.ones_like(self.SAK1.weight, dtype=torch.float32)/16.0)
        # self.SAK1.bias=nn.Parameter(torch.zeros_like(self.SAK1.bias, dtype=torch.float32))
        # self.SAK1.weight.requires_grad=False
        # self.SAK1.bias.requires_grad=False

        self.SAK2=nn.Conv2d(32,64,1,stride=2, bias=False)
        # self.SAK2.weight=nn.Parameter(torch.ones_like(self.SAK2.weight, dtype=torch.float32)/32.0)
        # self.SAK2.bias=nn.Parameter(torch.zeros_like(self.SAK2.bias, dtype=torch.float32))
        # self.SAK2.weight.requires_grad=False
        # self.SAK2.bias.requires_grad=False


        """
        Layer Details:
         6n+1 Conv2d layers, each followed by relu
         Final Conv2d followed by meanPool over all features (Global Average Pooling)
         Then, flatten
         Then, fully connected with 'r' outputs.
         Then, softmax.
        """
        if Normalizer!=layerNorm2d:
            NormalizerProps['num_features']=16
            if Normalizer==groupNorm2d:
                NormalizerProps['h']=32
                NormalizerProps['w']=32
        self.layers=['' for _ in range(3*(6*n+1)+1+1+1)]

        self.layers[0]=nn.Conv2d(3, 16, 3, padding=self.padding_info[(32,32,3,1)])
        self.layers[1]=Normalizer(**NormalizerProps)
        self.layers[2]=nn.ReLU(inplace=True)
        for i in range(3, 6*n+3, 3):
            self.layers[i]=nn.Conv2d(16, 16, 3, padding=self.padding_info[(32,32,3,1)])
            self.layers[i+1]=Normalizer(**NormalizerProps)
            self.layers[i+2]=nn.ReLU(inplace=True)


        if Normalizer!=layerNorm2d:
            NormalizerProps['num_features']=32
            if Normalizer==groupNorm2d:
                NormalizerProps['h']=16
                NormalizerProps['w']=16
        self.layers[6*n+3]=nn.Conv2d(16, 32, 3, stride=2, padding=self.padding_info[(32,16,3,2)])
        self.layers[6*n+4]=Normalizer(**NormalizerProps)
        self.layers[6*n+5]=nn.ReLU(inplace=True)
        for i in range(6*n+6, 12*n+3, 3):
            self.layers[i]=nn.Conv2d(32, 32, 3, padding=self.padding_info[(16,16,3,1)])
            self.layers[i+1]=Normalizer(**NormalizerProps)
            self.layers[i+2]=nn.ReLU(inplace=True)


        if Normalizer!=layerNorm2d:
            NormalizerProps['num_features']=64
            if Normalizer==groupNorm2d:
                NormalizerProps['h']=8
                NormalizerProps['w']=8
        self.layers[12*n+3]=nn.Conv2d(32, 64, 3, stride=2, padding=self.padding_info[(16,8,3,2)])
        self.layers[12*n+4]=Normalizer(**NormalizerProps)
        self.layers[12*n+5]=nn.ReLU(inplace=True)
        for i in range(12*n+6, 18*n+3, 3):
            self.layers[i]=nn.Conv2d(64, 64, 3, padding=self.padding_info[(8,8,3,1)])
            self.layers[i+1]=Normalizer(**NormalizerProps)
            self.layers[i+2]=nn.ReLU(inplace=True)

        # self.layers[12*n+2]=torch.mean()
        self.layers[18*n+3]=nn.Flatten()
        self.layers[18*n+4]=nn.Linear(64, self.num_classes)
        self.layers[18*n+5]=nn.Softmax(dim=0)

        # convert self.layers to nn.ModuleList
        self.layers=nn.ModuleList(self.layers)

        return

    def forward(self, x):
        n=int((len(self.layers)-6)/18)

        x=self.layers[2](self.layers[1](self.layers[0](x)))
        prev_to_add=x
        for i in range(3, 18*n+3, 6):
            x=self.layers[i+2](self.layers[i+1](self.layers[i](x)))
            if i==6*n+3:
                # prev_to_add has to be increased in depth and halved in height, width
                #  16x32x32 to 32x16x16
                prev_to_add=self.SAK1(prev_to_add)

            elif i==12*n+3:
                # prev_to_add has to be increased in depth and halved in height, width
                prev_to_add=self.SAK2(prev_to_add)
            else:
                continue
            x=self.layers[i+5](self.layers[i+4](self.layers[i+3](x)+prev_to_add))
            prev_to_add=x
        x=torch.mean(x, dim=(2,3))
        x=self.layers[18*n+3](
            self.layers[18*n+4](
                self.layers[18*n+5](x)
                )
            )
        return x

