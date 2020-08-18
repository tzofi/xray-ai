import torch
import torchvision
import torchvision.transforms as transforms

import csv
import os
import numpy as np
from math import floor, ceil
import scipy.ndimage as ndimage

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=2):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock5x5, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv5x5(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet7_2_1(nn.Module):

    def __init__(self, block, blocks_per_layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, add_softmax=False, latent_dim=768):
        super(ResNet7_2_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 8, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, blocks_per_layers[5], stride=2)
        self.layer7 = self._make_layer(block, 192, blocks_per_layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(latent_dim, output_channels)
        self.softmax = nn.Softmax(dim=1)
        self.add_softmax = add_softmax
        #self.fc1 = nn.Linear(768, 24)
        #self.bn2 = nn.BatchNorm1d(24)
        #self.fc2 = nn.Linear(24, output_channels) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        #x = self.avgpool(x)
        z = torch.flatten(x, 1)
        #x = self.fc1(z)
        #x = self.bn2(x)
        #x = self.relu(x)
        #y = self.fc2(x)
        y = self.fc1(z)
        if self.add_softmax:
            y = self.softmax(y)

        return y, z

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        
        # Saving path should be a directory where the model and configuration can be saved
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        # model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        print("Loading the model")
        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


class ResNet7_4_1(nn.Module):

    def __init__(self, block, blocks_per_layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, add_softmax=False, latent_dim=768):
        super(ResNet7_4_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 192, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 384, blocks_per_layers[5], stride=2)
        self.layer7 = self._make_layer(block, 768, blocks_per_layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc1 = nn.Linear(latent_dim, output_channels)
        self.softmax = nn.Softmax(dim=1)
        self.add_softmax = add_softmax
        #self.fc1 = nn.Linear(768, 24)
        #self.bn2 = nn.BatchNorm1d(24)
        #self.fc2 = nn.Linear(24, output_channels) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        z = self.layer7(x)
        
        x = self.avgpool(z)
        x = torch.flatten(x, 1)
        #x = self.fc1(z)
        #x = self.bn2(x)
        #x = self.relu(x)
        #y = self.fc2(x)
        y = self.fc1(x)
        if self.add_softmax:
        	y = self.softmax(y)
        	
        return y, z

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        
        # Saving path should be a directory where the model and configuration can be saved
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        # model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        print("Loading the model")
        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


class ResNet8_2_1(nn.Module):

    def __init__(self, block, blocks_per_layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, add_softmax=False, latent_dim=768):
        super(ResNet8_2_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 192, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 384, blocks_per_layers[5], stride=2)
        self.layer7 = self._make_layer(block, 768, blocks_per_layers[6], stride=2)
        self.layer8 = self._make_layer(block, 768, blocks_per_layers[7], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(latent_dim, output_channels)
        self.softmax = nn.Softmax(dim=1)
        self.add_softmax = add_softmax        
        #self.fc1 = nn.Linear(768, 24)
        #self.bn2 = nn.BatchNorm1d(24)
        #self.fc2 = nn.Linear(24, output_channels) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        z = self.layer8(x)

        x = self.avgpool(z)
        x = torch.flatten(x, 1)
        #x = self.fc1(z)
        #x = self.bn2(x)
        #x = self.relu(x)
        #y = self.fc2(x)
        y = self.fc1(x)
        if self.add_softmax:
        	y = self.softmax(y)        
        
        return y, z

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        
        # Saving path should be a directory where the model and configuration can be saved
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        # model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        print("Loading the model")
        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


class ResNet5_4_2(nn.Module):

    def __init__(self, block, blocks_per_layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, add_softmax=False, latent_dim=2048):
        super(ResNet5_4_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock5x5, 8, blocks_per_layers[0], stride=4)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_channels)
        self.softmax = nn.Softmax(dim=1)
        self.add_softmax = add_softmax        
        #self.fc1 = nn.Linear(768, 24)
        #self.bn2 = nn.BatchNorm1d(24)
        #self.fc2 = nn.Linear(24, output_channels) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        y = self.fc2(x)
        if self.add_softmax:
        	y = self.softmax(y)        
        
        return y, z

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        
        # Saving path should be a directory where the model and configuration can be saved
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        # model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers,
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        print("Loading the model")
        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


def resnet14_1(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2, 2], 
			   pretrained=False, checkpoint=None, latent_dim=768, **kwargs):
    model = ResNet7_2_1(block, blocks_per_layers, latent_dim=latent_dim, **kwargs)
    if pretrained:
        model = model.from_pretrained(checkpoint, block, blocks_per_layers, **kwargs)
    return model

def resnet14_16(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2, 2], 
			   pretrained=False, checkpoint=None, latent_dim=768, **kwargs):
    model = ResNet7_4_1(block, blocks_per_layers, latent_dim=latent_dim, **kwargs)
    if pretrained:
        model = model.from_pretrained(checkpoint, block, blocks_per_layers, **kwargs)
    return model

def resnet14_4(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2, 2, 2], 
			   pretrained=False, checkpoint=None, latent_dim=768, **kwargs):
    model = ResNet8_2_1(block, blocks_per_layers, latent_dim=latent_dim, **kwargs)
    if pretrained:
        model = model.from_pretrained(checkpoint, block, blocks_per_layers, **kwargs)
    return model

def resnet10_16(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2], 
			   pretrained=False, checkpoint=None, latent_dim=2048, **kwargs):
    model = ResNet5_4_2(block, blocks_per_layers, latent_dim=latent_dim, **kwargs)
    if pretrained:
        model = model.from_pretrained(checkpoint, block, blocks_per_layers, **kwargs)
    return model
