import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import clip
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer, OpenAIGPTTokenizer, OpenAIGPTModel
from torchvision.models import resnet18, resnet
from transformers.models.bert.modeling_bert import BertConfig
import functools
import copy
from transformers import AutoTokenizer, AutoModel

from .similarity_mining import MultilabelContrastiveLoss

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


@functools.lru_cache(maxsize=128)
def get_bert_stuff():
    BERT_MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=False)
    BERT_model = BertModel.from_pretrained(BERT_MODEL_NAME, local_files_only=False)
    return BERT_model, tokenizer

@functools.lru_cache(maxsize=128)
def get_distilbert_stuff():
    DistilBERT_model_name = "distilbert-base-uncased"
    DistilBERT_model = DistilBertModel.from_pretrained(DistilBERT_model_name, local_files_only=False)
    DistilBERT_tokenizer = DistilBertTokenizer.from_pretrained(DistilBERT_model_name, local_files_only=False)
    return DistilBERT_model, DistilBERT_tokenizer

@functools.lru_cache(maxsize=128)
def get_gpt1_stuff():
    model_name = 'openai-gpt'
    model = OpenAIGPTModel.from_pretrained(model_name, local_files_only=False)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, local_files_only=False)
    return model, tokenizer

@functools.lru_cache(maxsize=128)
def get_bge_stuff():
    model_name = "BAAI/bge-base-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = AutoModel.from_pretrained(model_name, local_files_only=False)
    return model, tokenizer


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

# adapted from
# https://github.com/VICO-UoE/DatasetCondensation
# https://github.com/Zasder3/train-CLIP


''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out



''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width=128, net_depth=4, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size = (224,224)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNetGAP(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNetGAP, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(shape_feat[0], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def ResNet18ImageNet(channel, num_classes):
    return ResNetImageNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet6ImageNet(channel, num_classes):
    return ResNetImageNet(BasicBlock, [1,1,1,1], channel=channel, num_classes=num_classes)


## Sourced directly from OpenAI's CLIP repo
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

import timm

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

class DropPath(nn.Module):
    def __init__(self, p=0.): super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0. or not self.training: return x
        keep = 1 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

# tmp
class ProjectionHeadV4(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int,
                 hidden_mult: int = 4, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, projection_dim)
        self.ln1 = nn.LayerNorm(projection_dim, eps=1e-5)
        self.fc1 = nn.Linear(projection_dim, projection_dim * hidden_mult * 2)
        self.fc1_out = nn.Linear(projection_dim * hidden_mult, projection_dim)
        self.dp1 = nn.Dropout(dropout)
        self.droppath1 = DropPath(drop_path)
        self.g1 = nn.Parameter(torch.ones(projection_dim) * 1e-5)
        self.ln2 = nn.LayerNorm(projection_dim, eps=1e-5)
        hidden_dim = projection_dim * hidden_mult * 2
        self.fc2 = nn.Linear(projection_dim, hidden_dim)
        self.fc2_out = nn.Linear(hidden_dim // 2, projection_dim)
        self.dp2 = nn.Dropout(dropout)
        self.droppath2 = DropPath(drop_path)
        self.g2 = nn.Parameter(torch.ones(projection_dim) * 1e-5)
        self.ln_out = nn.LayerNorm(projection_dim, eps=1e-5)
        self.act = GEGLU()

    def forward(self, x):
        x = self.proj(x)
        y = self.act(self.fc1(self.ln1(x)))
        y = self.fc1_out(y)
        y = self.droppath1(self.dp1(y))
        x = x + self.g1 * y
        y = self.act(self.fc2(self.ln2(x)))
        y = self.fc2_out(y)
        y = self.droppath2(self.dp2(y))
        x = x + self.g2 * y
        return self.ln_out(x)


@functools.lru_cache(maxsize=128)
def load_from_timm(model_name, pretrained):
    if model_name == 'clip':
        raise NotImplementedError("it is unfair to use pretrained clip")
        # if pretrained:
        #     model, preprocess = clip.load("ViT-B/32", device='cuda')
        # else:
        #     configuration = ViTConfig()
        #     model = ViTModel(configuration)

    elif model_name == 'nfnet':
        model = timm.create_model('nfnet_l0', pretrained=pretrained, num_classes=0, global_pool="avg")
    elif model_name == 'vit':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    elif model_name == 'nf_resnet50':
        model = timm.create_model('nf_resnet50', pretrained=True)
    elif model_name == 'nf_regnet':
        model = timm.create_model('nf_regnet_b1', pretrained=True)
    elif model_name=="efficientvit_m5":
        model = timm.create_model(model_name, num_classes=0, pretrained=True)
    elif model_name=="dinov2":
        model = timm.create_model("vit_base_patch16_224.dino", num_classes=0, pretrained=True)
    else:
        model = timm.create_model(model_name, num_classes=0, pretrained=True, global_pool="avg")


    return model
    



class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, args):
        super().__init__()
        self.model_name = args.image_encoder
        self.pretrained = args.image_pretrained
        self.trainable = args.image_trainable

        self.model = copy.deepcopy(load_from_timm(self.model_name, self.pretrained))
        # use cached pretrained model

        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        if self.model_name == 'clip' and self.pretrained:
            return self.model.encode_image(x)
        else:
            return self.model(x)
            
    def gradient(self, x, y):
        # Compute the gradient of the mean squared error loss with respect to the weights
        loss = self.loss(x, y)
        grad = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        return torch.cat([g.view(-1) for g in grad])




class TextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pretrained = args.text_pretrained
        self.trainable = args.text_trainable
        self.model_name = args.text_encoder
        
        if self.model_name == 'clip':
            self.model, preprocess = clip.load("ViT-B/32", device='cuda')
        elif self.model_name == 'bert':
            pt_model, self.tokenizer = get_bert_stuff()
            if args.text_pretrained:
                self.model = pt_model
            else:
                self.model = BertModel(BertConfig())
                self.model.init_weights()
        elif self.model_name == 'distilbert':
            self.model, self.tokenizer = get_distilbert_stuff()
        elif self.model_name == 'gpt1':
            self.model, self.tokenizer = get_gpt1_stuff()
        elif self.model_name == 'bge':
            self.model, self.tokenizer = get_bge_stuff()
        else:
            raise NotImplementedError(self.model_name)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
    
    def forward(self, texts, device='cuda'):
        if self.model_name == 'clip':
            output = self.model.encode_text(clip.tokenize(texts).to('cuda'))  

        elif self.model_name == 'bert':
            # Tokenize the input text
            encoding = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, self.target_token_idx, :]
        
        elif self.model_name == 'distilbert':
            encoding = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, self.target_token_idx, :]

        elif self.model_name == 'gpt1':
            self.tokenizer.pad_token = ' '
            encoding = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, self.target_token_idx, :]

        elif self.model_name == 'bge':
            encoding = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = self.model(input_ids, attention_mask=attention_mask).last_hidden_state[:, self.target_token_idx, :]

        return output


class CLIPModel_full(nn.Module):
    def __init__(
        self,
        args,
        train_logit_scale=False,
        temperature=0.07,
    ):
        super().__init__()

        self.out_embedding = args.out_embedding
        
        if args.image_encoder == 'nfnet':
            self.image_embedding = 2304
        elif args.image_encoder == 'convnet':
            self.image_embedding = 768
        elif args.image_encoder == 'resnet18':
            self.image_embedding = 512
        elif args.image_encoder == 'convnext':
            self.image_embedding = 640
        elif args.image_encoder == 'dinov2':
            self.image_embedding = 768
        else:
            self.image_embedding = 1000

        if args.text_encoder == 'clip':
            self.text_embedding = 512 
        elif args.text_encoder == 'bert':
            self.text_embedding = 768 
        elif args.text_encoder == 'distilbert':
            self.text_embedding = 768
        elif args.text_encoder == 'gpt1':
            self.text_embedding = 768
        elif args.text_encoder == 'bge':
            self.text_embedding = 768
        else:
            raise NotImplementedError
        
        self.image_encoder =  ImageEncoder(args)
        self.text_encoder = TextEncoder(args)
        
        if args.image_trainable:
            self.text_projection = ProjectionHead(embedding_dim=self.text_embedding, projection_dim=self.image_embedding).to(args.device)
        else:
            self.text_projection = ProjectionHead(embedding_dim=self.text_embedding, projection_dim=self.out_embedding).to(args.device)
            self.image_projection = ProjectionHead(embedding_dim=self.image_embedding, projection_dim=self.out_embedding).to(args.device)
            

        if train_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))
        else:
            self.logit_scale = torch.ones([]) * np.log(1.0 / temperature)

        self.image_trainable = args.image_trainable
        self.distill = args.distill

        self.multilabel_criterion = MultilabelContrastiveLoss(args.loss_type)


    def forward(self, image, caption, similarity=None):
        self.image_encoder = self.image_encoder.to('cuda')
        self.text_encoder = self.text_encoder.to('cuda')
        
        image_features = self.image_encoder(image)
        text_features = caption if self.distill else self.text_encoder(caption)

        if self.image_trainable:
            combined_image_features = image_features.float()
        else:
            combined_image_features = self.image_projection(image_features.float())
        combined_text_features = self.text_projection(text_features.float())
        image_features = combined_image_features / combined_image_features.norm(dim=1, keepdim=True)
        text_features = combined_text_features / combined_text_features.norm(dim=1, keepdim=True)

        image_logits = self.logit_scale.exp() * image_features @ text_features.t() 

        ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
        acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum().item()
        acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum().item()
        acc = (acc_i + acc_t) / 2


        if similarity is None:
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
        else:
            loss = self.multilabel_criterion(image_logits, similarity)

        return loss, acc
        
class SampleNet(nn.Module):
    """
    TNet module for adversarial networks with fixed activation layers and predefined parameters.
    """

    def __init__(self, feature_dim=64, t_batchsize=64, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim  # Feature dimension
        self.t_sigma_num = t_batchsize // 16  # Number of sigmas for t_net
        self._input_adv_t_net_dim = feature_dim  # Input noise dimension
        self._input_t_dim = feature_dim  # t_net input dimension
        self._input_t_batchsize = t_batchsize  # Batch size
        self._input_t_var = t_var  # Variance of input noise

        # Fixed activation layers
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # Create a simple 3-layer fully connected network using fixed activation layers
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self, device):
        # Generate white noise
        if self.t_sigma_num > 0:
            # Initialize the white noise input
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var**0.5)
            self._t_net_input = self._t_net_input.to(device).detach()

            # Forward pass
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)

            # Generate the final t value
            # self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * ((self._input_t_var / self._input_t_dim) ** 0.5)
            # self._t = self._t.to(device).detach()
            self._t = a
        else:
            # When t_sigma_num = 0, generate standard Gaussian noise as t
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device).detach()
        return self._t
