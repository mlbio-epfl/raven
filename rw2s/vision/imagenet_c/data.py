""" From github.com/openai/weak-to-strong.git """
import torch
import torchvision

### ImageNet
IMAGENET_RESIZE, IMAGENET_CROP = 256, 224
IMAGENET_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGENET_RESIZE),
    torchvision.transforms.CenterCrop(IMAGENET_CROP),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_imagenet(datapath, split, batch_size, shuffle, transform=IMAGENET_TRANSFORM):
    dset = torchvision.datasets.ImageNet(root=datapath, split=split, transform=transform)
    dl = torch.utils.data.DataLoader(dset, shuffle=shuffle, batch_size=batch_size, num_workers=1)
    return dset, dl


### ImageNet-C (github.com/hendrycks/robustness.git)
IMAGENET_C_RESIZE, IMAGENET_C_CROP = 256, 224
IMAGENET_C_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGENET_C_RESIZE),
    torchvision.transforms.CenterCrop(IMAGENET_C_CROP),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_imagenet_c(datapath, batch_size, shuffle, transform=IMAGENET_C_TRANSFORM):
    dset = torchvision.datasets.ImageFolder(root=datapath, transform=transform)
    dl = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dset, dl
