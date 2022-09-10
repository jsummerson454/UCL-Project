import torch
import torch.nn as nn

# Faithful implementation of the PatchGAN discriminator described in https://arxiv.org/abs/1611.07004
# PatchGAN is a confusing name, it is actually just a fully convolutional network, but with
# the depth and parameters configured such that the receptive field of the outputs are
# 70x70, and these are averaged to give the overall prediction

# bias stuff same as before, no bias if using batchnorm (which we are) or AN AFFINE instancenorm

# It is perfectly reasonable to use a regular fully colvolutional network that reduces down to
# a single value, it is still a valid discriminator model, just not a PatchGAN

class PatchGAN(nn.Module):    
    # note in_channels is 6 by default as it concatenates both the input and conditional images (assumed
    # to both be 3 by default, so default is set to 6)
    def __init__(self, in_channels=6, norm_layer=nn.InstanceNorm2d):
        super(PatchGAN, self).__init__()
         
        self.model = nn.Sequential(
            # First layer is somewhat non-regular - it doesn't include normalisation
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
        
            # Regular layers
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
            
            # notice stride change for this semi-final layer
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),
                                   
            # final layer maps to 1 dimensional output (1 channel) and has stride 1 like the previous
            # this time DOES include bias due to no normalisation after
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True),
        )
        
        
    # needs BOTH images (conditional and output)
    def forward(self, composite, polyp):
        x = torch.cat([composite, polyp], 1)
        return self.model(x)