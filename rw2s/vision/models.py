"""
From github.com/openai/weak-to-strong.git
"""
import torch
from torch import nn
import torchvision


def get_model(name, device, pretrained=True, replace_last_layer_with_n_classes=None):
    if name == "alexnet":
        model = alexnet(pretrained=pretrained, replace_last_layer_with_n_classes=replace_last_layer_with_n_classes)
    elif name == "squeezenetv1_1":
        model = squeezenetv1_1(pretrained=pretrained, replace_last_layer_with_n_classes=replace_last_layer_with_n_classes)
    elif name == "resnet18":
        model = resnet18(pretrained=pretrained, replace_last_layer_with_n_classes=replace_last_layer_with_n_classes)
    elif name == "resnet50":
        model = resnet50(pretrained=pretrained, replace_last_layer_with_n_classes=replace_last_layer_with_n_classes)
    elif name == "densenet121":
        model = densenet121(pretrained=pretrained, replace_last_layer_with_n_classes=replace_last_layer_with_n_classes)
    elif name == "resnet50_dino":
        assert pretrained
        model = resnet50_dino()
    elif name == "vitb8_dino":
        assert pretrained
        model = vitb8_dino()
        if replace_last_layer_with_n_classes is not None:
            print("[INFO] Adding a linear layer to DINO Vit-B/8")
            model = LinearProbeClassifier(
                backbone=model,
                classifier=nn.Linear(model.embed_dim, replace_last_layer_with_n_classes),
            )
    elif name == "vits8_dino":
        assert pretrained
        model = vits8_dino()
        if replace_last_layer_with_n_classes is not None:
            print("[INFO] Adding a linear layer to DINO Vit-S/8")
            model = LinearProbeClassifier(
                backbone=model,
                classifier=nn.Linear(model.embed_dim, replace_last_layer_with_n_classes),
            )
    elif name == "vitl14_dinov2":
        assert pretrained
        model = vitl14_dinov2()
        if replace_last_layer_with_n_classes is not None:
            print("[INFO] Adding a linear layer to DINOv2 Vit-L/14 distilled")
            model = LinearProbeClassifier(
                backbone=model,
                classifier=nn.Linear(model.embed_dim, replace_last_layer_with_n_classes),
            )
    else:
        raise ValueError(f"Unknown model {name}")
    model.to(device)
    model.eval()
    model = nn.DataParallel(model, device_ids=[device])
    return model


def freeze_backbone(model, model_name):
    if model_name == "densenet121":
        for param in model.parameters():
            param.requires_grad = False
        ### unfreeze the 'classifier' part of the model
        for param in model.module.classifier.parameters():
            param.requires_grad = True
    elif model_name == "alexnet":
        for param in model.parameters():
            param.requires_grad = False
        ### unfreeze the last layer of the classifier
        for param in model.module.classifier.head[-1].parameters():
            param.requires_grad = True
    elif model_name == "squeezenetv1_1":
        for param in model.parameters():
            param.requires_grad = False
        ### unfreeze the last layer of the classifier
        for param in model.module.model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "resnet18" or model_name == "resnet50":
        for param in model.parameters():
            param.requires_grad = False
        ### unfreeze the last layer of the classifier
        for param in model.module.fc.parameters():
            param.requires_grad = True
    elif model_name == "resnet50_dino":
        raise NotImplementedError("Freezing all but last layer not implemented for DINO models")
    elif model_name == "vitb8_dino":
        assert hasattr(model, "module") and hasattr(model.module, "classifier"), "Model does not have a classifier"
        for param in model.parameters():
            param.requires_grad = False
        ### unfreeze the the classifier
        for param in model.module.classifier.parameters():
            param.requires_grad = True
    elif model_name == "vits8_dino":
        assert hasattr(model, "module") and hasattr(model.module, "classifier"), "Model does not have a classifier"
        for param in model.parameters():
            param.requires_grad = False
        ### unfreeze the the classifier
        for param in model.module.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model {model_name}")

    return model


class HeadAndEmbedding(torch.nn.Module):
    def __init__(self, head):
        super(HeadAndEmbedding, self).__init__()
        self.head = head

    def forward(self, x):
        return x, self.head(x)


class HeadAndEmbeddingSqueezeNet(torch.nn.Module):
    def __init__(self, model):
        super(HeadAndEmbeddingSqueezeNet, self).__init__()
        self.model = model

    def forward(self, x):
        emb = self.model.features(x)
        return torch.flatten(emb, 1), torch.flatten(self.model.classifier(emb), 1)


class LinearProbeClassifier(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super(LinearProbeClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        emb = self.backbone(x)
        return emb, self.classifier(emb)


def _alexnet_replace_fc(model):
    model.classifier = HeadAndEmbedding(model.classifier)
    return model


def _squeezenet_replace_fc(model):
    model = HeadAndEmbeddingSqueezeNet(model)
    return model


def _resnet_replace_fc(model):
    model.fc = HeadAndEmbedding(model.fc)
    return model


def _densenet_replace_fc(model):
    model.classifier = HeadAndEmbedding(model.classifier)
    return model


def resnet50_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    return model


def vitb8_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    return model


def vits8_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    return model


def vitl14_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    return model


def alexnet(pretrained=True, replace_last_layer_with_n_classes=None, get_latents=True):
    if pretrained:
        model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    else:
        model = torchvision.models.alexnet()

    ### replace last classification layer
    if replace_last_layer_with_n_classes is not None:
        model.classifier[-1] = nn.Linear(
            in_features=model.classifier[-1].in_features,
            out_features=replace_last_layer_with_n_classes,
            bias=model.classifier[-1].bias is not None,
        )

    ### get also the latent representations from the model
    if get_latents:
        model = _alexnet_replace_fc(model)

    return model


def squeezenetv1_1(pretrained=True, replace_last_layer_with_n_classes=None, get_latents=True):
    if pretrained:
        model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
    else:
        model = torchvision.models.squeezenet1_1()

    ### replace last classification layer
    if replace_last_layer_with_n_classes is not None:
        model.classifier[1] = nn.Conv2d(
            in_channels=model.classifier[1].in_channels,
            out_channels=replace_last_layer_with_n_classes,
            kernel_size=model.classifier[1].kernel_size,
        )

    ### get also the latent representations from the model
    if get_latents:
        model = _squeezenet_replace_fc(model)

    return model


def resnet18(pretrained=True, replace_last_layer_with_n_classes=None, get_latents=True):
    if pretrained:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    else:
        model = torchvision.models.resnet18()

    ### replace last classification layer
    if replace_last_layer_with_n_classes is not None:
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=replace_last_layer_with_n_classes,
            bias=model.fc.bias is not None,
        )

    ### get also the latent representations from the model
    if get_latents:
        model = _resnet_replace_fc(model)

    return model


def resnet50(pretrained=True, replace_last_layer_with_n_classes=None, get_latents=True):
    if pretrained:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.resnet50()

    ### replace last classification layer
    if replace_last_layer_with_n_classes is not None:
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=replace_last_layer_with_n_classes,
            bias=model.fc.bias is not None,
        )

    ### get also the latent representations from the model
    if get_latents:
        model = _resnet_replace_fc(model)

    return model


def densenet121(pretrained=True, replace_last_layer_with_n_classes=None, get_latents=True):
    if pretrained:
        model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
    else:
        model = torchvision.models.densenet121()

    ### replace last classification layer
    if replace_last_layer_with_n_classes is not None:
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features,
            out_features=replace_last_layer_with_n_classes,
            bias=model.classifier.bias is not None,
        )

    ### get also the latent representations from the model
    if get_latents:
        model = _densenet_replace_fc(model)

    return model

