import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

from rw2s.vision.imagenet_c.data import get_imagenet_c, IMAGENET_C_TRANSFORM


TFORMS = {
    "iwildcam": transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.3838, 0.3879, 0.3602], [0.2594, 0.2592, 0.2587]),
    ]),
    "camelyon17": transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.7440, 0.5895, 0.7213], [0.1787, 0.2131, 0.1721]),
    ]),
    "fmow": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4159, 0.4184, 0.3914], [0.2448, 0.2408, 0.2436]),
    ]),
    "imagenet": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "imagenet_c": IMAGENET_C_TRANSFORM,
}


def get_imagenet(datapath, split, batch_size, shuffle, transform, num_workers=1, class_range=None):
    print("Getting ImageNet data...", split)
    dset = torchvision.datasets.ImageNet(root=datapath, split=split, transform=transform)
    if class_range is not None:
        start, end = class_range
        targets = np.array(dset.targets)
        mask = (targets >= start) & (targets <= end)
        sample_indices = np.where(mask)[0]
        dset = torch.utils.data.Subset(dset, sample_indices)
        dset.classes = dset.dataset.classes
        dset_classes_filtered = dset.dataset.classes[start:end + 1]
    dl = torch.utils.data.DataLoader(dset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dset, dl


def get_mean_std(dl):
    """ Modified from source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/Basics/pytorch_std_mean.py """

    ### var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for imgs, _, _ in tqdm(dl):
        channels_sum += torch.mean(imgs, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(imgs**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std


def filter_dset_by_metadata(dset, meta_key, meta_start_end_idxs=None, meta_start_end_val_range=None):
    assert not (meta_start_end_idxs is not None and meta_start_end_val_range is not None)
    meta_key_idx = dset.metadata_fields.index(meta_key)

    if meta_start_end_idxs is not None:
        ### filter by idx range
        meta_start_end_val_range = dset.metadata_array[:, meta_key_idx].unique(sorted=True)[meta_start_end_idxs[0]:meta_start_end_idxs[1]]

    ### filter by value range
    filter_in_idxs = np.where(
        (dset.metadata_array[:, meta_key_idx] >= meta_start_end_val_range[0])
        & (dset.metadata_array[:, meta_key_idx] <= meta_start_end_val_range[-1])
    )[0]

    return WILDSSubset(dataset=dset, indices=filter_in_idxs, transform=None), meta_start_end_val_range


def filter_dl_by_metadata(dataloader, meta_key, meta_start_end_idxs=None, meta_start_end_val_range=None):
    dset_filtered, new_meta_start_end_val_range = filter_dset_by_metadata(dset=dataloader.dataset, meta_key=meta_key, meta_start_end_idxs=meta_start_end_idxs, meta_start_end_val_range=meta_start_end_val_range)
    dl = get_train_loader("standard", dset_filtered, batch_size=dataloader.batch_size, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory)
    return dl, new_meta_start_end_val_range


def get_dataloader(dset, tier, cfg):
    if tier == "train":
        dl = get_train_loader("standard", dset, batch_size=cfg["batch_size"], num_workers=cfg["n_threads"], pin_memory=True)
    elif tier in ("id_val", "val", "test"):
        dl = get_eval_loader("standard", dset, batch_size=cfg["batch_size"], num_workers=cfg["n_threads"], pin_memory=True)
    else:
        raise ValueError(f"Data tier {tier} not recognized.")
    return dl


def get_data(cfg):
    assert cfg["name"] in TFORMS, f"Transforms for dataset called {cfg['name']} not found."
    tform = TFORMS[cfg["name"]]
    dsets, dls = dict(), dict()

    if cfg["name"] == "imagenet":
        for tier in ("train", "val"):
            dsets[tier], dls[tier] = get_imagenet(
                datapath=os.path.join(cfg["path"], "imagenet"),
                split=tier,
                batch_size=cfg["batch_size"],
                shuffle=tier == "train",
                transform=tform,
                num_workers=cfg["n_threads"],
                class_range=cfg.get(f"{tier}_class_range", None),
            )

            ### also make a subsplit dataloader
            if cfg.get(f"subsplit_{tier}", None) is not None:
                assert len(cfg[f"subsplit_{tier}"]) == 2 and sum(cfg[f"subsplit_{tier}"]) == 1.
                dsets[f"{tier}_split0"], dsets[f"{tier}_split1"] = torch.utils.data.random_split(dsets[tier], lengths=cfg[f"subsplit_{tier}"], generator=torch.Generator().manual_seed(cfg.get("seed", 0)))
                dls[f"{tier}_split0"] = torch.utils.data.DataLoader(dsets[f"{tier}_split0"], shuffle=tier == "train", batch_size=cfg["batch_size"], num_workers=cfg["n_threads"])
                dls[f"{tier}_split1"] = torch.utils.data.DataLoader(dsets[f"{tier}_split1"], shuffle=tier == "train", batch_size=cfg["batch_size"], num_workers=cfg["n_threads"])
    elif cfg["name"] == "imagenet_c":
        assert "type" in cfg, "Please specify the corruption type."

        dsets["test"], dls["test"] = get_imagenet_c(
            datapath=os.path.join(cfg["path"], cfg["name"], cfg["type"]),
            batch_size=cfg["batch_size"],
            shuffle=False,
            transform=tform,
        )
    else: # WILDS
        ### get data splits
        dset = get_dataset(dataset=cfg["name"], download=False, root_dir=os.path.join(cfg["path"], cfg["name"]))
        for tier in ("train", "id_val", "val", "test"):
            dsets[tier] = dset.get_subset(tier, transform=tform)
            dls[tier] = get_dataloader(dset=dsets[tier], tier=tier, cfg=cfg)

            ### also make a subsplit dataloader
            if cfg.get(f"subsplit_{tier}", None) is not None:
                ### subsplit this tier
                assert len(cfg[f"subsplit_{tier}"]) == 2 and sum(cfg[f"subsplit_{tier}"]) == 1.
                dsets[f"{tier}_split0"] = dset.get_subset(tier, frac=cfg[f"subsplit_{tier}"][0], transform=tform)
                all_tier_idxs = np.where(dset.split_array == dset.split_dict[tier])[0]
                split0_idxs = dsets[f"{tier}_split0"].indices
                split1_idxs = np.array(list(set(all_tier_idxs).difference(set(split0_idxs))))
                dsets[f"{tier}_split1"] = WILDSSubset(dataset=dset, indices=split1_idxs, transform=tform)

                dls[f"{tier}_split0"] = get_dataloader(dset=dsets[f"{tier}_split0"], tier=tier, cfg=cfg)
                dls[f"{tier}_split1"] = get_dataloader(dset=dsets[f"{tier}_split1"], tier=tier, cfg=cfg)

    ### optionally get the mean and std
    # mean, std = get_mean_std(dls["train"])
    # print(f"{mean=}\n{std=}")

    return dsets, dls
