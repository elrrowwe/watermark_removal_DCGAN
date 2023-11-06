import torch.nn as nn
import torch.nn.functional as F

"""
This file contains the generator part of my GAN model. 
"""

class Generator(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=3):
        """
        This method initializes the generator part of the GAN model.
        ngf: the size of feature maps in the generator
        nc: the number of input channels (3 for RGB images)
        """
        super(Generator, self).__init__() # call the super constructor
        self.G = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(3, ngf * 8, 3, 2, 0, bias=False),
            # nn.LayerNorm(ngf * 8),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True), # an inplace ReLU activation function
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            # nn.LayerNorm(ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            # nn.LayerNorm(ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, bias=False),
            # nn.LayerNorm(ngf),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 3, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``       
            )

    def forward(self, z):
        """
        The forward pass of the generator.
        """
        return self.G(z)