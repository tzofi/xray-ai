import torch
import torch.nn as nn
import torchvision.models as models

from torch.distributions.normal import Normal

def same_padding(kernel_size, dilation):
    """Same padding for nn.ZeroPad2d"""
    padding = int(0.5 * dilation * (kernel_size - 1))
    return padding


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=True)


def tconv3x3(in_planes, out_planes, stride=1):
    """3x3 tranpose convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=True)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, bias=True)


def conv7x7(in_planes, out_planes, stride=1):
    """7x7 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, bias=True)


def sampling(mu, logvar):
    """Sampling using reparameterization trick"""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn_like(sigma)
    return mu + eps*sigma


class TransposeLayer(nn.Module):
    """Keras Style Tranpose Layer: ConvTranpose2d + ReLU"""
    def __init__(self, in_planes, out_planes, conv=tconv3x3, stride=2):
        super(TransposeLayer, self).__init__()
        self.conv = conv(in_planes, out_planes, stride=stride)
        self.pad = nn.ZeroPad2d(same_padding(self.conv.kernel_size[0], 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class ResLayer(nn.Module):
    """Basic ResNet Layer"""
    def __init__(self, in_planes, out_planes, conv=conv1x1, stride=1, bn=True, activation=True):
        super(ResLayer, self).__init__()
        self.bn = bn
        self.activation = activation
        self.conv = conv(in_planes, out_planes, stride=stride)
        self.pad = nn.ZeroPad2d(same_padding(self.conv.kernel_size[0], 1))
        if self.bn: self.bn = nn.BatchNorm2d(out_planes)
        if self.activation: self.activation = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        if self.bn: out = self.bn(out)
        if self.activation: out = self.activation(out)
        return out


class ResBlock(nn.Module):
    """Block of ResNet Layers"""
    def __init__(self, in_planes, out_planes, conv=conv1x1, stride=1, conv_to_skip=False):
        super(ResBlock, self).__init__()
        self.conv_to_skip = conv_to_skip
        self.layer1 = ResLayer(in_planes, out_planes, conv=conv, stride=stride, bn=True, activation=True)
        self.layer2 = ResLayer(out_planes, out_planes, conv=conv, stride=1, bn=True, activation=False)
        if self.conv_to_skip:
            self.skip_conv = ResLayer(in_planes, out_planes, conv=conv1x1, stride=stride, bn=False, activation=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if self.conv_to_skip:
            x = self.skip_conv(x)
        out = x + out
        out = self.relu(out)
        return out


class ResStack(nn.Module):
    """Stack of ResNet Blocks"""
    def __init__(self, in_planes, out_planes, conv=conv3x3, first_stride=2, stride=1, num_res_blocks=2):
        super(ResStack, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            if i == 0: self.blocks.append(ResBlock(in_planes, out_planes, conv=conv, stride=first_stride, conv_to_skip=True))
            else: self.blocks.append(ResBlock(out_planes, out_planes, conv=conv, stride=stride, conv_to_skip=False))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        

class VEncoder(nn.Module):
    """Variational Encoder built from ResNet Stacks"""
    def __init__(self, in_planes, out_planes=128, filters=8, num_res_stacks=5, num_res_blocks=2):
        super(VEncoder, self).__init__()
        self.first_layer = ResLayer(in_planes, filters, conv=conv7x7, stride=4)
        self.stacks = nn.ModuleList()
        in_planes = filters
        for i in range(num_res_stacks):
            if i == 0:
                self.stacks.append(ResStack(in_planes, filters, conv=conv5x5, first_stride=2, stride=1, num_res_blocks=2))
            else:
                self.stacks.append(ResStack(in_planes, filters, conv=conv3x3, first_stride=2, stride=1, num_res_blocks=2))
            in_planes = filters
            filters = 2 * filters
        self.last_layer = ResLayer(in_planes, out_planes, conv=conv1x1, stride=1, bn=False, activation=False)

    def forward(self, x):
        x = self.first_layer(x)
        for stack in self.stacks:
            x = stack(x)
        x = self.last_layer(x)
        return x


class VDecoder(nn.Module):
    """Variational Decoder built from Transpose Layers"""
    def __init__(self, in_shape=[256, 8, 8], num_stacks=8, conv=tconv3x3, stride=2):
        super(VDecoder, self).__init__()
        self.in_shape = in_shape
        self.stacks = nn.ModuleList()
        in_planes = in_shape[0]
        for i in range(num_stacks):
            # Only difference between this architecture and original Keras implementation is kernel size of the transpose convolution: we use 2 instead of 3.
            self.stacks.append(TransposeLayer(in_planes, in_planes//2, conv=conv, stride=stride))
            in_planes = in_planes//2

    def forward(self, x):
        x = x.view(-1, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        for stack in self.stacks:
            x = stack(x)
        return x


class VAE(nn.Module):
    """VAE Implementation for ChestXRay Images"""
    def __init__(self, input_dim, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = VEncoder(input_dim, out_planes=z_dim)
        self.decoder = VDecoder()

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        z = sampling(mu, logvar)
        out = self.decoder(z)
        return out, z, mu, logvar


class AE(nn.Module):
    """AE Implementation for ChestXRay Images"""
    def __init__(self, input_dim, z_dim):
        super(AE, self).__init__()
        self.z_dim = z_dim
        self.encoder = VEncoder(input_dim, out_planes=z_dim)
        self.decoder = VDecoder(in_shape=[128,16,16], num_stacks=7)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class Regressor(nn.Module):
    """Regressor for edema quantification"""
    def __init__(self, in_shape=[1, 128, 128], filters=8, num_res_stacks=5, num_res_blocks=2, num_classes=3):
        super(Regressor, self).__init__()
        self.in_shape = in_shape
        in_planes = self.in_shape[0]
        self.stacks = nn.ModuleList()
        for i in range(num_res_stacks):
            self.stacks.append(ResStack(in_planes, filters, conv=conv3x3, first_stride=2, stride=1, num_res_blocks=2))
            in_planes = filters
            filters = 2 * filters
        self.pool = nn.AvgPool2d(2)
        #self.linear1 = nn.Linear(512, 64)
        self.linear1 = nn.Linear(128*32*32, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        for stack in self.stacks:
            x = stack(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class ResRegressor(nn.Module):
    """Resnet to Regressor for edema quantification"""
    def __init__(self, in_shape=[1, 128, 128], filters=8, num_res_stacks=2, num_res_blocks=2, num_classes=3, final_activation="sigmoid"):
        super(ResRegressor, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        #self.features = nn.Sequential(*list(resnet.children())[:-4])
        #for param in self.features:
        #    param.requires_grad = False
        self.in_shape = in_shape
        in_planes = self.in_shape[0]
        self.stacks = nn.ModuleList()
        for i in range(num_res_stacks):
            self.stacks.append(ResStack(in_planes, filters, conv=conv3x3, first_stride=2, stride=1, num_res_blocks=2))
            in_planes = filters
            filters = 2 * filters
        self.pool = nn.AvgPool2d(2)
        #self.linear1 = nn.Linear(512, 64)
        #self.linear1 = nn.Linear(128*4*4, 64)
        self.linear1 = nn.Linear(64*1*1, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, num_classes)
        if final_activation == "sigmoid":
            self.final = nn.Sigmoid()
        elif final_activation == "softmax":
            self.final = nn.Softmax() 

    def forward(self, x):
        #x = torch.cat((x, x, x), 1)
        #x = self.features(x)
        x = x.view(-1, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        for stack in self.stacks:
            x = stack(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        if self.final != None:
            x = self.final(x)
        return x


class VRegressor(nn.Module):
    """VAE to generate latent Z followed by quantification by regressor"""
    def __init__(self, input_dim=1, z_dim=128):
        super(VRegressor, self).__init__()
        self.VAE = VAE(input_dim, z_dim)
        self.regressor = Regressor()

    def forward(self, x, vae_only=False):
        if vae_only:
            out, z, mu, logvar = self.VAE(x)
            return out, z, mu, logvar
        else:
            out, z, mu, logvar = self.VAE(x)
            q = self.regressor(z)
            return q, out, z, mu, logvar
        '''
        mu, logvar = self.VAE.encoder(x).chunk(2, dim=1)
        z = sampling(mu, logvar)
        q = self.regressor(z)
        return q, z, mu, logvar
        '''
