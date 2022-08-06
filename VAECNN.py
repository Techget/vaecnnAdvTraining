'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

def sample_image(mu, log_var):
    # log_var = self.conv1_log_var(x)
    #     mu = self.conv1_mu(x)
    std = torch.exp(torch.mul(log_var, 0.5))
    eps = torch.randn_like(std)
    return eps * std + mu

class VAEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(VAEBasicBlock, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.conv1_mu =  nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv1_log_var =  nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2_mu = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2_log_var = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.conv_mu_shortcut = nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            self.conv_logvar_shortcut = nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        mu1 = self.conv1_mu(x)
        log_var1 = self.conv1_log_var(x)
        std1 = torch.exp(torch.mul(log_var1, 0.5))
        eps1 = torch.randn_like(std1)
        conv1 = eps1 * std1 + mu1
        out = F.relu(self.bn1(conv1))

        log_var2 = self.conv2_log_var(out)
        mu2 = self.conv2_mu(out)
        std2 = torch.exp(torch.mul(log_var2, 0.5))
        eps2 = torch.randn_like(std2)
        conv2 = eps2 * std2 + mu2
        out = self.bn2(conv2)

        # out += self.shortcut(x)
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            mu_shortcut = self.conv_mu_shortcut(x)
            logvar_shortcut = self.conv_logvar_shortcut(x)
            std_shortcut  = torch.exp(torch.mul(logvar_shortcut, 0.5))
            eps_shortcut = torch.randn_like(std_shortcut)
            conv_shortcut = eps_shortcut * std_shortcut + mu_shortcut
            out += self.bn_shortcut(conv_shortcut)
        else:
            out += self.shortcut(x)

        out = F.relu(out)
        return out, kl_loss(mu=mu1, log_var=log_var1) + kl_loss(mu=mu2, log_var=log_var2)

class VAEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(VAEResNet, self).__init__()
        self.in_planes = 64

        self.conv1_mu =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv1_log_var =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # for stride in strides:
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def kl_loss(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

    def forward(self, x, _eval=False):
        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        log_var = self.conv1_log_var(x)
        mu = self.conv1_mu(x)
        std = torch.exp(torch.mul(log_var, 0.5))
        eps = torch.randn_like(std)
        conv1 = eps * std + mu
        out = F.relu(self.bn1(conv1))

        out, kl_loss1 = self.layer1(out)
        out, kl_loss2 = self.layer2(out)
        out, kl_loss3 = self.layer3(out)
        out, kl_loss4 = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        self.train()
        
        return [out, self.kl_loss(mu, log_var) + kl_loss1 + kl_loss2 + kl_loss3 + kl_loss4]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class VAECNNFirstLayerChanged(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(VAECNNFirstLayerChanged, self).__init__()
        self.in_planes = 64

        self.conv1_mu =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv1_log_var =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, _eval=False):
        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        log_var = self.conv1_log_var(x)
        mu = self.conv1_mu(x)
        std = torch.exp(torch.mul(log_var, 0.5))
        eps = torch.randn_like(std)
        conv1 = eps * std + mu
        out = F.relu(self.bn1(conv1))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        self.train()

        return out , kl_loss(mu, log_var)


def VAEResNet18FirstLayerChanged():
    return VAECNNFirstLayerChanged(BasicBlock, [2,2,2,2])

def VAEResNet18():
    return VAEResNet(VAEBasicBlock, [2,2,2,2])
