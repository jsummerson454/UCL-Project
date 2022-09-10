import torch
import torch.nn as nn

# A fairly faithful reimplementation of the pix2pix generator unet as described in https://arxiv.org/abs/1611.07004

# Biases of conv layers are only included if they are not immediately followed by a normalisation layer

# Decreasing layers of unet - Convolution-batchnorm-ReLU layers, convolutions are 4x4 with stride 2 and
# padding 1 hence downsampling by a factor of 2
# For encoder layers use leaky ReLU (with slope 0.2), for decoder layers non-leaky ReLU
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, normalise=True, norm_layer=nn.InstanceNorm2d):
        super(EncoderLayer, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=(not normalise))]
        if normalise:
            layers.append(norm_layer(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.combined = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.combined(x)
    
# Increasing layers of unet - Convolution-batchnorm-ReLU as before, 4x4, stride 2, padding 1, but now using
# transpose convolutions for the upsampling and skip connections
class DecoderLayerWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, normalise=True, innermost=False, norm_layer=nn.InstanceNorm2d):
        super(DecoderLayerWithSkip, self).__init__()
        
        # flag to handle innermost case - if this is set then there is no skip connection available
        if not innermost:
            in_channels = in_channels*2
            
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=(not normalise))]
        if normalise:
            layers.append(norm_layer(out_channels))
        layers.append(nn.ReLU())
        self.combined = nn.Sequential(*layers)
        
    # skip connection required as argument to forward operator, feed it the output of the corresponding
    # layer in the encoder
    def forward(self, x, skip_connection):
        return torch.cat([skip_connection, self.combined(x)], 1)
        
        
class Pix2PixGenerator(nn.Module):
    # 8 layers reduces 256x256 image to 1x1 bottleneck
    # Any exceptions to layers (such as normalisation included or not) are mentioned in comments beforehand
    # all components are stored (self.) so that they are returned by .children() and hence can be initalised
    # easier using apply(), due to how the forward operator and module as a whole is constructed.
    def __init__(self, in_channels=3, out_channels=3, norm_layer=nn.InstanceNorm2d):
        super(Pix2PixGenerator, self).__init__()

        # --- ENCODER ---
        
        # First exception - for whatever reason the first layer in the encoder does not have normalisation applied
        # This is represented in the code and mentioned in the appendix of the paper itself, so is intentional
        self.enc1 = EncoderLayer(in_channels, 64, normalise=False, norm_layer=norm_layer)
        self.enc2 = EncoderLayer(64, 128, norm_layer=norm_layer)
        self.enc3 = EncoderLayer(128, 256, norm_layer=norm_layer)
        self.enc4 = EncoderLayer(256, 512, norm_layer=norm_layer)
        self.enc5 = EncoderLayer(512, 512, norm_layer=norm_layer)
        self.enc6 = EncoderLayer(512, 512, norm_layer=norm_layer)
        self.enc7 = EncoderLayer(512, 512, norm_layer=norm_layer)
        # Another exception, no normalisation in final layer of encoder as it would zero out the bottleneck with
        # a batch size of 1
        self.enc8 = EncoderLayer(512, 512, normalise = False, norm_layer=norm_layer)
        
        self.encoder = [self.enc8, self.enc7, self.enc6, self.enc5, self.enc4, self.enc3, self.enc2, self.enc1]
        
        
        # --- DECODER ---
        
        # Another exception, no skip connection on bottleneck layer (nothing to skip!) so set
        # the innermost flag to control this edge case handling
        self.dec8 = DecoderLayerWithSkip(512, 512, innermost=True, norm_layer=norm_layer)
        self.dec7 = DecoderLayerWithSkip(512, 512, norm_layer=norm_layer)
        self.dec6 = DecoderLayerWithSkip(512, 512, norm_layer=norm_layer)
        self.dec5 = DecoderLayerWithSkip(512, 512, norm_layer=norm_layer)
        self.dec4 = DecoderLayerWithSkip(512, 256, norm_layer=norm_layer)
        self.dec3 = DecoderLayerWithSkip(256, 128, norm_layer=norm_layer)
        self.dec2 = DecoderLayerWithSkip(128, 64, norm_layer=norm_layer)
        # Last exception, final layer of encoder is completely different as it produces the output image
        # No normalisation, and applies a tanh activation (input channels doubled due to skip connection
        # from previous decoder layer) --- NO NORMALISATION => BIAS IS INCLUDED
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )
        
        self.decoder = [self.dec8, self.dec7, self.dec6, self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]
        
        assert len(self.encoder) == len(self.decoder)
        self.depth = len(self.encoder)
        
    # Given the result of the decoder at the layer above i (i=0 at innermost layer) compute the output of
    # layer i (warning - this notation for layers is opposite to the one used for defining them in __init__)
    # Note that for the last layer the decoder does not need the skip connection input to append to it's results
    # and so must be handled in a separate case
    def recurse(self, dec, i):
        down = self.encoder[i](dec)
        
        # base case
        if i==0:
            up = self.decoder[i](down, dec)
            
        # special case for final decoder layer
        elif i==self.depth-1:
            up = self.decoder[i](self.recurse(down, i-1))
            
        # general case
        else:
            up = self.decoder[i](self.recurse(down, i-1), dec)
        
        return up
            

    def forward(self, x):    
        # Apply the encoder and decoder layers in a recursive manner from the top down
        return self.recurse(x, self.depth-1)


# A fairly faithful reimplementation of the CycleGAN generator (9 resnet block version) as described in https://arxiv.org/pdf/1703.10593.pdf
class ResnetBlock(nn.Module):
    def __init__(self, channels, norm_layer=nn.InstanceNorm2d):
        super(ResnetBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            norm_layer(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True), # bias=False if ReLU after...
            norm_layer(channels),
            #nn.ReLU(True) 
        )
     
    def forward(self, x):
        out = x + self.block(x) # SKIP CONNECTIONS
        return out
        
class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm_layer=nn.InstanceNorm2d):
        super(CycleGANGenerator, self).__init__()
        
        # Initial 7x7 convolution
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=False),
                 norm_layer(64),
                 nn.ReLU(True)]
        
        # downsampling layers x2
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(128),
                  nn.ReLU(True),
                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                  norm_layer(128),
                  nn.ReLU(True)]
        
        # resnet blocks x9
        for i in range(9):
            model += [ResnetBlock(256, norm_layer=norm_layer)]
            
        # upsampling layers x2
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  norm_layer(128),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  norm_layer(64),
                  nn.ReLU(True)]
        
        # Final 7x7 convolution
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_channels, kernel_size=7, padding=0, bias=True),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# Custom generator architecture A - uses the concatenated dilated convolution approach seen in the edge polyp synthesis model
class DilationBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 rate1_channels,
                 rate2_channels,
                 rate4_channels,
                 out_channels,
                 norm_layer=nn.InstanceNorm2d):

        super(DilationBlock, self).__init__()

        self.rate1 = nn.Conv2d(in_channels, rate1_channels, kernel_size=3, padding=1, dilation=1)
        self.rate2 = nn.Conv2d(in_channels, rate2_channels, kernel_size=3, padding=2, dilation=2)
        self.rate4 = nn.Conv2d(in_channels, rate4_channels, kernel_size=3, padding=4, dilation=4)
        
        channel_sum = rate1_channels + rate2_channels + rate4_channels
        self.final = nn.Conv2d(channel_sum, out_channels, kernel_size=1, padding=0, bias=False)
        
        self.normalisation = norm_layer(channel_sum)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        r1 = self.activation(self.rate1(x))
        r2 = self.activation(self.rate2(x))
        r4 = self.activation(self.rate4(x))
        
        combined = torch.cat((r1,r2,r4), 1)
        
        out = self.activation(self.normalisation(self.final(combined)))
        return out
    
# see Implementation chapter of dissertation for more detailed explanation
class CustomGeneratorA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm_layer=nn.InstanceNorm2d):
        super(CustomGeneratorA, self).__init__()
        
        # initial 7x7 instancenormed convolution for contrast invariance
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=0, bias=False),
            norm_layer(32),
            nn.ReLU()
        )
        
        # First dilation block into downsampling
        self.dil1 = DilationBlock(32, 16, 16, 32, 32, norm_layer=norm_layer)
        self.down1 = EncoderLayer(32, 64, norm_layer=norm_layer)
        
        # Second dilation block into downsampling
        self.dil2 = DilationBlock(64, 32, 32, 64, 64, norm_layer=norm_layer)
        self.down2 = EncoderLayer(64, 128, norm_layer=norm_layer)
        
        # Third and final dilation block into downsampling
        self.dil3 = DilationBlock(128, 64, 64, 128, 128, norm_layer=norm_layer)
        self.down3 = EncoderLayer(128, 256, norm_layer=norm_layer)
        
        resblocks = []
        for i in range(6):
            resblocks += [ResnetBlock(256, norm_layer=norm_layer)]
            
        self.resblock_section = nn.Sequential(*resblocks)
        
        # Upconvolution 1
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU()
        )
        
        # Upconvolution 2 - NOTE DOUBLED INPUT CHANNELS DUE TO SKIP CONNECTIONS IN UNET
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU()
        )
        
        # Upconvolution 3 - NOTE DOUBLED INPUT CHANNELS DUE TO SKIP CONNECTIONS IN UNET
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU()
        )
        
        # Final 7x7 convolution to produce output image
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0, bias=True),
            nn.Tanh()
        )
        
    def forward(self, x):
        # apply initial convolution
        x = self.initial(x)
        
        # apply each of the dilation blocks into downsampling layers, saving the outputs for skip connection usage
        d1 = self.dil1(x)
        d2 = self.dil2(self.down1(d1))
        d3 = self.dil3(self.down2(d2))
        
        # bottleneck layer output
        bottleneck = self.resblock_section(self.down3(d3))
        
        # concatenate with skip connections whilst upsampling
        u3 = self.upconv1(bottleneck)
        c3 = torch.cat((u3, d3), 1)
        u2 = self.upconv2(c3)
        c2 = torch.cat((u2, d2), 1)
        u1 = self.upconv3(c2)
        c1 = torch.cat((u1, d1), 1)
        
        # final convolution which maps to output channels
        out = self.final(c1)
        return out
        

# Custom architecture B - hybrid model of pix2pix unet and cyclegan
# does 3 downsampling layers (256^2 -> 32^2 bottleneck) using same encoder details as pix2pix, then
# some resnet blocks (as done in cyclegan), then upsamples WITH SKIP CONNECTIONS
class CustomGeneratorB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm_layer=nn.InstanceNorm2d):
        super(CustomGeneratorB, self).__init__()

        # Borrowing the initial 7x7 convolution used in cyclegan
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=0, bias=False),
            norm_layer(32),
            nn.ReLU()
        )

        # initial 3 encoder layers
        self.enc1 = EncoderLayer(32, 64, norm_layer=norm_layer)
        self.enc2 = EncoderLayer(64, 128, norm_layer=norm_layer)
        self.enc3 = EncoderLayer(128, 256, norm_layer=norm_layer)

        # resnet bottleneck
        resnet_blocks = []
        for i in range(6):
            resnet_blocks += [ResnetBlock(256, norm_layer=norm_layer)]
        self.bottleneck_layer = nn.Sequential(*resnet_blocks)

        # decoder layers (upsampling layers consist of resize-convolution blocks)
        # (note indexing is inverse order they are applied - index of LAYER)
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU()
        )

        # Inputs doubled for rest of network due to skip connections
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU()
        )

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU()
        )

        # Final 7x7 convolution to produce output image
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        # Non-bottleneck encoder layers are stored (including the full resolution map
        # map after the initial 7x7 convolution) for skip connection usage
        s1 = self.initial(x)
        s2 = self.enc1(s1)
        s3 = self.enc2(s2)
        bottleneck = self.enc3(s3)

        bottleneck = self.bottleneck_layer(bottleneck)

        u3 = self.upconv3(bottleneck)
        u3 = torch.cat((u3, s3), 1)
        u2 = self.upconv2(u3)
        u2 = torch.cat((u2, s2), 1)
        u1 = self.upconv1(u2)
        u1 = torch.cat((u1, s1), 1)

        out = self.final(u1)
        return out
