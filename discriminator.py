import torch.nn as nn
import torch.nn.functional as F

"""
This file contains the discriminator part of the network.
"""

class Discriminator(nn.Module):
    def __init__(self, ndf=128, nc=3):
        """
        This method initializes the discriminator part of the GAN model.
        ndf: the size of feature maps in the discriminator
        nc: the number of input channels (3 for RGB images)
        """
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            # the input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.17, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            # nn.LayerNorm(ndf * 2),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.17, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.LayerNorm(ndf * 4),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.17, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            # nn.LayerNorm(ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.17, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, inp):
        """
        The forward pass of the discriminator.
        """
        return self.D(inp)