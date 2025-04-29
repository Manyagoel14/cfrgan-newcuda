import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet50
from model.networks import VGG19
# from utils import load_state_dict
from torch.autograd import Variable


class PerceptualLoss(nn.Module):
    def __init__(self, model_type='resnet'):
        super(PerceptualLoss, self).__init__()
        if model_type=='resnet':
            self.criterion = ResLoss()
        elif model_type=='vgg':
            self.criterion = VGGLoss()
        else:
            raise ValueError('Model type should be one of resnet or vgg')
    
    def forward(self, y_hat, y):
        if not y_hat.is_cuda:
            y_hat = y_hat.cuda()
        if not y.is_cuda:
            y = y.cuda()
        return self.criterion(y_hat, y)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        if not x.is_cuda:
            x = x.cuda()
        if not y.is_cuda:
            y = y.cuda()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class ResLoss(nn.Module):
    def __init__(self):
        super(ResLoss, self).__init__()
        self.criterion = nn.L1Loss()
        resnet = resnet50(pretrained=True)
        if torch.cuda.is_available():
            resnet = resnet.cuda()

        self.B1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        if torch.cuda.is_available():
            self.B1 = self.B1.cuda()

    def forward(self, _data, _target):
        if not _data.is_cuda:
            _data = _data.cuda()
        if not _target.is_cuda:
            _target = _target.cuda()
        data, target = self.B1(_data), self.B1(_target)
        B1_loss = torch.mean(torch.square(data - target), dim=[1,2,3])

        data, target = self.B2(data), self.B2(target)
        B2_loss = torch.mean(torch.square(data - target), dim=[1,2,3])

        data, target = self.B3(data), self.B3(target)
        B3_loss = torch.mean(torch.square(data - target), dim=[1,2,3])

        return B1_loss + B2_loss + B3_loss


class WGAN_DIV_Loss(nn.Module):
    def __init__(self, k=2, p=6, dim=289):
        super(WGAN_DIV_Loss, self).__init__()
        self.k = k
        self.p = p
        self.dim = dim
    
    def forward(self, real_val, y, fake_val, y_hat):
        real_grad_out = Variable(torch.cuda.FloatTensor(real_val.size(0), self.dim).fill_(1.0), requires_grad=False)
        real_grad = torch.autograd.grad(
            real_val, y, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (self.p/2)

        fake_grad_out = Variable(torch.cuda.FloatTensor(fake_val.size(0), self.dim).fill_(0.0), requires_grad=False)
        fake_grad = torch.autograd.grad(
            fake_val, y_hat, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (self.p/2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * self.k / 2

        # WGAN-DIV loss
        loss_D = -torch.mean(real_val) + torch.mean(fake_val) + div_gp
        # Spectral normalization loss
        # loss_D = torch.mean(F.relu(-1 + real_val)) + torch.mean(F.relu(-1-fake_val))
    
        return loss_D
    

