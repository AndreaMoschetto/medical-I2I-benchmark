import functools
from torch import nn
import torch


class UnetSkipConnectionBlock2D(nn.Module):
    def __init__(self, outer_nc, inner_nc, in_channels=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=True):
        super(UnetSkipConnectionBlock2D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:  # noqa
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if in_channels is None:
            in_channels = outer_nc
        downconv = nn.Conv2d(in_channels, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UNetGenerator2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_downs=7, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True):
        super(UNetGenerator2D, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock2D(ngf * 8, ngf * 8, in_channels=None, submodule=None, norm_layer=norm_layer, innermost=True)  # innermost
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock2D(ngf * 8, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock2D(ngf * 4, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2D(ngf * 2, ngf * 4, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2D(ngf, ngf * 2, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock2D(out_channels, ngf, in_channels=in_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator2D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator2D, self).__init__()
        if type(norm_layer) == functools.partial:  # noqa # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  # kernel width
        padw = 1  # padding width
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
