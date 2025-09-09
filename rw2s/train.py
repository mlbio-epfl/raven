import os
from copy import deepcopy
import numpy as np
import tqdm
import dill
import torch
from functools import partial
from datetime import datetime

from rw2s.utils import seed_all, preload
from rw2s.losses import LOSS_DICT


def train_head(
    teacher_model,
    student_model,
    dataloader,
    cfg,
    logger,
    cached_labels_path,
    cached_embs_path,
    results,
    rng,
    n_classes,
    return_data=False,
    additional_eval_data=None,
    before_optim_run_callback_weak=None,
    before_optim_run_callback_gt=None,
    after_batch_callback_weak=None,
    before_batch_callback_weak=None,
    after_batch_callback_gt=None,
    before_batch_callback_gt=None,
):
    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        # load from cache
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        gt_labels, teacher_labels, teacher_acc = cached["gt_labels"], cached["teacher_labels"], cached["teacher_acc"]
    else:
        # collect (and save)
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels...")
        teacher_embeddings, gt_labels, teacher_labels, teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=chunking_dir)
        if cfg["w2s"]["save_labels"]:
            torch.save({
                "cfg": cfg,
                "embeddings": teacher_embeddings,
                "inps": None,
                "gt_labels": gt_labels,
                "teacher_labels": teacher_labels,
                "teacher_acc": teacher_acc,
            }, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        # load from cache
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        student_embeddings, inps, student_gt_labels = cached["embeddings"], cached["inps"], cached["gt_labels"]
    else:
        # collect (and save)
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info(f"Collecting embeddings (chunking directory: {chunking_dir})...")
        student_embeddings, student_gt_labels, student_labels, student_acc, inps, _ = preload(model=student_model, loader=dataloader, device=cfg["device"], chunking_dir=chunking_dir, store_embs=True)
        if cfg["w2s"]["save_embeddings"]:
            torch.save({
                "cfg": cfg,
                "embeddings": student_embeddings,
                "inps": inps,
                "gt_labels": student_gt_labels,
                "student_labels": student_labels,
                "student_acc": student_acc,
            }, cached_embs_path, pickle_module=dill)
    assert torch.all(gt_labels == student_gt_labels), "GT labels from teacher and student do not match."
    del student_gt_labels

    ### order of samples
    order = np.arange(len(gt_labels))
    rng.shuffle(order)
    results["order"].append(order)

    ### all data
    x = student_embeddings[order]
    y = gt_labels[order]
    yw = teacher_labels[order]

    ### split
    assert len(cfg["w2s"]["train_val_test_split"]) == 3, "Train, val, test split must be of length 3."
    assert sum(cfg["w2s"]["train_val_test_split"]) == 1.0, "Train, val, test split must sum to 1."
    n_train, n_val = int(cfg["w2s"]["train_val_test_split"][0] * len(x)), int(cfg["w2s"]["train_val_test_split"][1] * len(x))
    x_train, x_val, x_test = x[:n_train], x[n_train:n_train+n_val], x[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    yw_train, yw_val, yw_test = yw[:n_train], yw[n_train:n_train+n_val], yw[n_train+n_val:]
    yw_val = (yw_val.mean(1) if yw_val.ndim == 3 else yw_val).argmax(-1) # only for evaluation
    yw_test = (yw_test.mean(1) if yw_test.ndim == 3 else yw_test).argmax(-1) # only for evaluation
    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "test": (x_test, y_test), "test_weak": (x_test, yw_test)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v
    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples: {len(x_train)}.")
    logger.info(f"  Number of validation samples: {len(x_val)}.")
    logger.info(f"  Number of testing samples: {len(x_test)}.")

    ### eval teacher (average weak labels)
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)
    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")

    ### w2s
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"]) # important to get same results for cached/not cached
    results_teacher_to_student, student_model_probe = train_logreg(x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"], loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_weak, after_batch_callback=after_batch_callback_weak)
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=n_train + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, _ = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe


def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
):
    ### setup training data
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(
        x_train,
        y_train,
        torch.arange(len(y_train)),
        sample_weights if sample_weights is not None else torch.ones(len(y_train), device=device),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    ### setup model and optimizer
    model = torch.nn.Linear(x_train.shape[-1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        ### train
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.step()
            schedule.step()

            ### calc metrics
            if len(y.shape) == 2:
                y = torch.argmax(y, dim=-1)
            elif len(y.shape) == 3:
                y = y.mean(1).argmax(-1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}")

        ### eval
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2:
                    if not warning_printed:
                        print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                        warning_printed = True
                    pred = pred.mean(1)
                if len(pred.shape) > 1:
                    pred = torch.argmax(pred, dim=-1)
                if len(y_test.shape) > 1:
                    y_test = torch.argmax(y_test, dim=-1)
                acc = (pred == y_test).float().mean()
                results[f"{key}_all"].append(acc)

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    return results, model


def train(
    model,
    train_dl,
    loss_fn,
    val_dl=None,
    val_loss_fn=None,
    weights=None, # torch.Tensor of shape (N,)
    normalize_weights_per_batch=False,
    n_epochs=30,
    optimizer_name="Adam",
    optimizer_kwargs=dict(),
    scheduler_name=None,
    scheduler_mul_factor=None,
    early_stopping_patience=None,
    load_best_model=True,
    logger=None,
    wdb_run=None,
    ckpt_path=None,
    device=0,
):
    assert (early_stopping_patience is None and not load_best_model) or val_dl is not None, \
        "Validation dataloader is required for early stopping."
    assert weights is None or (weights.ndim == 1 and len(weights) == len(train_dl.dataset)), \
        "Weights must be a 1D tensor of length equal to the number of training samples."

    ### setup optimization and tracking
    opter = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_kwargs)
    n_iter = (n_batches := len(train_dl)) * n_epochs
    iter_i = 0
    if scheduler_name == None:
        schedule = None
    elif scheduler_name == "cosine":
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opter, T_max=n_iter)
    elif scheduler_name == "multiplicative":
        schedule = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=opter, lr_lambda=lambda ep: scheduler_mul_factor)
    else:
        raise ValueError(f"Scheduler name {scheduler_name} not recognized.")
    val_loss_fn = deepcopy(loss_fn) if val_loss_fn is None else val_loss_fn
    best = {"val_loss": np.inf, "val_acc": 0}
    ea_worse_epochs = 0

    ### load if ckpt exists
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)
        model.load_state_dict(ckpt["model"])
        opter.load_state_dict(ckpt["opter"])
        if scheduler_name is not None and ckpt["scheduler"] is not None:
            schedule.load_state_dict(ckpt["scheduler"])
        epoch = ckpt["epoch"]
        iter_i = ckpt["iter_i"]
        best = ckpt["best"]
        logger.info(f"Loaded checkpoint from epoch {epoch}.")

    ### run
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        correct, total, loss_ep = 0, 0, 0

        model.train()
        for b in train_dl:
            x, y = b[0].to(device), b[1].to(device)

            opter.zero_grad()
            pred = model(x)
            if len(pred) == 2 and type(pred) is not torch.Tensor:
                pred = pred[1]
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, reduction="none")

            ### weight loss
            if weights is not None:
                batch_weights = weights[total:total+len(y)]
                if normalize_weights_per_batch:
                    batch_weights = batch_weights / batch_weights.sum()
                loss = loss * batch_weights

            ### backprop
            loss.mean().backward()
            opter.step()
            if scheduler_name == "cosine":
                schedule.step()

            ### logging
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            loss_ep += loss.sum().item()

        ### eval
        val_correct, val_total, val_loss = 0, 0, 0
        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                for b in val_dl:
                    x, y = b[0].to(device), b[1].to(device)
                    pred = model(x)
                    if len(pred) == 2 and type(pred) is not torch.Tensor:
                        pred = pred[1]
                    loss = val_loss_fn(pred, y, step_frac=None, reduction="sum")

                    ### logging
                    if len(y.shape) > 1:
                        y = torch.argmax(y, dim=1)
                    val_correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
                    val_loss += loss.item()
                    val_total += len(y)

            val_loss_ep = val_loss / val_total
            val_acc = val_correct / val_total
            if wdb_run is not None:
                wdb_run.log({"val_loss": val_loss_ep, "val_acc": val_acc}, commit=False)

            ### update best
            if val_acc > best["val_acc"]:
                best["model"] = deepcopy(model.state_dict())
                best["val_loss"] = val_loss_ep
                best["val_acc"] = val_acc
                best["epoch"] = epoch
                ea_worse_epochs = 0
            else:
                ea_worse_epochs += 1

        ### logging
        if wdb_run is not None:
            wdb_run.log({"train_loss": loss_ep / total, "train_acc": correct / total})
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.4f}, Val Acc {round(val_correct / val_total, 4) if val_dl is not None else '---'}")
        logger.info(f"[{epoch}/{n_epochs}]  Train loss: {loss_ep / total:.4f}  |  Train acc: {correct / total:.4f}" +  (f"  |  Val loss: {val_loss_ep:.4f}  |  Val acc: {val_acc:.4f}" if val_dl is not None else ""))

        ### early stopping
        if early_stopping_patience is not None and ea_worse_epochs > early_stopping_patience:
            pbar.set_description(f"Early stopping. Best val loss: {best['val_loss']:.4f}. Best val acc: {best['val_acc']:.4f}.")
            logger.info(f"Early stopping. Best val loss: {best['val_loss']:.4f}. Best val acc: {best['val_acc']:.4f}.")
            break

        if scheduler_name == "multiplicative":
            schedule.step()

        ### save checkpoint
        if ckpt_path is not None:
            torch.save({
                "model": model.state_dict(),
                "opter": opter.state_dict(),
                "scheduler": schedule.state_dict() if schedule is not None else None,
                "epoch": epoch,
                "iter_i": iter_i,
                "best": best,
            }, ckpt_path, pickle_module=dill)

    ### load best model
    if load_best_model and "model" in best:
        if logger is not None:
            logger.info(f"Loading the best model from epoch {best['epoch']}. Validation loss: {best['val_loss']:.4f}. Validation accuracy: {best['val_acc']:.4f}.")
        model.load_state_dict(best["model"])

    return model
