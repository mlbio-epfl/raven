"""
From github.com/openai/weak-to-strong.git
"""
import os
import sys
import fire
import numpy as np
from datetime import datetime
import tqdm
import dill
from loguru import logger
import torch
from torch import nn

from rw2s.utils import seed_all, slugify
from rw2s.vision.imagenet_c.data import get_imagenet, get_imagenet_c
from rw2s.vision.imagenet_c.models import alexnet, resnet50_dino, vitb8_dino



def get_model(name, device):
    if name == "alexnet":
        model = alexnet()
    elif name == "resnet50_dino":
        model = resnet50_dino()
    elif name == "vitb8_dino":
        model = vitb8_dino()
    else:
        raise ValueError(f"Unknown model {name}")
    model.to(device)
    model.eval()
    # model = nn.DataParallel(model, device_ids=[device])
    return model


@torch.no_grad()
def get_embeddings(model, loader, device):
    all_embeddings, all_y, all_probs = [], [], []

    for x, y in tqdm.tqdm(loader):
        embeddings = model(x.to(device))
        if len(embeddings) == 2:
            embeddings, logits = embeddings
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
            all_probs.append(probs)

        all_embeddings.append(embeddings.detach().cpu())
        all_y.append(y)

    all_embeddings = torch.cat(all_embeddings, axis=0)
    all_y = torch.cat(all_y, axis=0)
    if len(all_probs) > 0:
        all_probs = torch.cat(all_probs, axis=0)
        acc = (torch.argmax(all_probs, dim=1) == all_y).float().mean()
    else:
        all_probs = None
        acc = None
    return all_embeddings, all_y, all_probs, acc


def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    device,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
):
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    d = x_train.shape[1]
    model = torch.nn.Linear(d, n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            schedule.step()
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}")

        for key, (x_test, y_test) in eval_datasets.items():
            x_test = x_test.float().to(device)
            pred = torch.argmax(model(x_test), axis=-1).detach().cpu()
            acc = (pred == y_test).float().mean()
            results[f"{key}_all"].append(acc)

    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    return results


def main(
    weak_model_name: str = "alexnet",
    strong_model_name: str = "resnet50_dino",
    batch_size: int = 128,
    n_train: int = 40_000,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 0,
    data_path: str = "/mlbio_scratch/sobotka/data/imagenet_c",
    save_path: str = "/mlbio_scratch/sobotka/w2s/imagenet_c",
    data_name: str = "gaussian_noise/3",
    save_cache: bool = True,
    load_cache: bool = True,
    device: int = 0,
):
    ### reproducibility
    seed_all(seed)
    rng = np.random.default_rng(seed)

    ### saving and logging
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_path = os.path.join(save_path, "cache")
    os.makedirs(cache_path, exist_ok=True)
    results_dir = os.path.join(save_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="{time:DD-MM HH:mm:ss} | {message}")

    ### prepare models and data
    logger.info("Loading models...")
    weak_model = get_model(name=weak_model_name, device=device)
    strong_model = get_model(name=strong_model_name, device=device)
    if "imagenet_c" in data_path:
        logger.info("Loading ImageNet-C...")
        _, dl = get_imagenet_c(datapath=os.path.join(data_path, data_name), batch_size=batch_size, shuffle=False)
    elif "imagenet" in data_path:
        logger.info("Loading ImageNet...")
        _, dl = get_imagenet(datapath=data_path, split="val", batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Couldn't match the specified `data_path` to a suitable data-loading function (Use 'imagenet_c' or 'imagenet' in `data_path`).")


    ### get weak labels
    cached_labels_path = os.path.join(cache_path, f"weak__{weak_model_name}__{slugify(data_name)}.pt")
    if load_cache and os.path.exists(cached_labels_path):
        # load from cache
        logger.info("Loading weak labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill)
        gt_labels, weak_labels, weak_acc = cached["gt_labels"], cached["weak_labels"], cached["weak_acc"]
    else:
        # collect (and save)
        logger.info("Collecting weak labels...")
        _, gt_labels, weak_labels, weak_acc = get_embeddings(model=weak_model, loader=dl, device=device)
        if save_cache:
            torch.save({
                "model_name": weak_model_name,
                "gt_labels": gt_labels,
                "weak_labels": weak_labels,
                "weak_acc": weak_acc,
                "run_name": run_name,
            }, cached_labels_path, pickle_module=dill)
    del weak_model # not needed anymore
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    logger.info(f"Weak label accuracy: {weak_acc:.3f}")
    
    ### get embeddings from strong model
    cached_embs_path = os.path.join(cache_path, f"strong__{strong_model_name}__{slugify(data_name)}.pt")
    if load_cache and os.path.exists(cached_embs_path):
        # load from cache
        logger.info("Loading strong model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        embeddings, strong_gt_labels = cached["embeddings"], cached["gt_labels"]
    else:
        # collect (and save)
        logger.info("Collecting strong embeddings...")
        embeddings, strong_gt_labels, _, _ = get_embeddings(model=strong_model, loader=dl, device=device)
        if save_cache:
            torch.save({
                "model_name": strong_model_name,
                "embeddings": embeddings,
                "gt_labels": strong_gt_labels,
                "run_name": run_name,
            }, cached_embs_path, pickle_module=dill)
    assert torch.all(gt_labels == strong_gt_labels)
    del strong_gt_labels

    ### prepare train and test for w2s training
    order = np.arange(len(embeddings))
    rng.shuffle(order)
    x = embeddings[order]
    y = gt_labels[order]
    yw = weak_labels[order]
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    yw_train, yw_test = yw[:n_train], yw[n_train:]
    yw_test = torch.argmax(yw_test, dim=1)
    eval_datasets = {"test": (x_test, y_test), "test_weak": (x_test, yw_test)}
    logger.info(f"Total number of samples: {len(x)}.")
    logger.info(f"Number of training samples: {len(x_train)}.")
    logger.info(f"Number of testing samples: {len(x_test)}.")

    seed_all(seed) # important to get same results for cached/not cached

    ### eval weak only on eval data
    weak_acc_test = (y_test == yw_test).float().mean()

    ### w2s
    logger.info("Training logreg on weak labels...")
    results_weak = train_logreg(x_train, yw_train, eval_datasets, device=device, batch_size=batch_size, n_epochs=n_epochs, lr=lr)
    logger.info(f"Final accuracy: {results_weak['test']:.3f}")
    logger.info(f"Final supervisor-student agreement: {results_weak['test_weak']:.3f}")
    logger.info(f"Accuracy by epoch: {[acc.item() for acc in results_weak['test_all']]}")
    logger.info(f"Supervisor-student agreement by epoch: {[acc.item() for acc in results_weak['test_weak_all']]}")

    ### gt labels
    logger.info("Training logreg on ground truth labels...")
    results_gt = train_logreg(x_train, y_train, eval_datasets, device=device, batch_size=batch_size, n_epochs=n_epochs, lr=lr)
    logger.info(f"Final accuracy: {results_gt['test']:.3f}")
    logger.info(f"Accuracy by epoch: {[acc.item() for acc in results_gt['test_all']]}")

    ### log results
    logger.info("\n\n" + "=" * 100)
    logger.info(f"Weak label accuracy (all data): {weak_acc:.3f}")
    logger.info(f"Weak label accuracy (test data): {weak_acc_test:.3f}")
    logger.info(f"Weak→Strong accuracy: {results_weak['test']:.3f}")
    logger.info(f"Strong accuracy: {results_gt['test']:.3f}")
    logger.info("=" * 100)

    ### save results
    results_path = os.path.join(results_dir, f"{run_name}__{slugify(data_name)}.pt")
    torch.save({
        "weak_model_name": weak_model_name,
        "strong_model_name": strong_model_name,
        "data_name": data_name,
        "batch_size": batch_size,
        "n_train": n_train,
        "n_epochs": n_epochs,
        "lr": lr,
        "seed": seed,
        "weak_acc": weak_acc,
        "weak_acc_test": weak_acc_test,
        "results_weak": results_weak,
        "results_gt": results_gt,
    }, results_path, pickle_module=dill)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    fire.Fire(main)
