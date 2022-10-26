import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import net_sphere

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def resnet18(outd=1, weights='IMAGENET1K_V1'):
    net = getattr(models, 'resnet18')(weights)
    net.fc = nn.Linear(512, outd)
    # net.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, outd))
    for name, param in net.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    return net


def sphere20a(outd=1, weights='runs/casian/model_best.pth.tar'):
    net = getattr(net_sphere, 'sphere20a')()
    net.fc6 = nn.Linear(512, 1)

  
    for name, param in net.named_parameters():
        if name not in ['fc6.weight', 'fc6.bias']:
            param.requires_grad = False

    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint['state_dict']
    # delete the last fc layer
    for param in list(state_dict.keys()):
        if param.startswith('module.'):
            if param not in ['module.fc6.weight']:
                state_dict[param[len("module."):]] = state_dict[param]
        del state_dict[param]

    msg = net.load_state_dict(state_dict, strict=False)
    print(msg.missing_keys)
    assert set(msg.missing_keys) == {'fc6.weight', 'fc6.bias'}

    return net

