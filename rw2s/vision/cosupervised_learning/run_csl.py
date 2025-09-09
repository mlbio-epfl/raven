""" Original code from https://github.com/YuejiangLIU/csl.git (modified)
@article{liu2024csl,
  author  = {Yuejiang Liu and Alexandre Alahi},
  title   = {Co-Supervised Learning: Improving Weak-to-Strong Generalization with Hierarchical Mixture of Experts},
  journal = {arXiv preprint 2402.15505},
  year    = {2024},
}
"""

import os
import sys
import fire
import numpy as np
import torch
from torch import nn
import tqdm
from ruamel.yaml import YAML
import json
from datetime import datetime
import dill
from collections import defaultdict
import lovely_tensors as lt

from rw2s.utils import seed_all, slugify
from rw2s.vision.data import get_data

lt.monkey_patch()


def get_conservative_estimate(y_next, y_prev, maxk=2):
    _, pred = y_prev.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_next.argmax(dim=1).view(1, -1).expand_as(pred))
    correct = correct.t()
    consensus = correct.any(dim=1)
    return consensus


def get_student_rank(logit_student, y_next):
    lw_train = nn.functional.cross_entropy(logit_student, y_next, reduction='none')
    lw_order = torch.argsort(lw_train) # ascending order
    lw_ranks = torch.argsort(lw_order)
    return lw_ranks


def get_output(model, x_train, n_classes=1000, batch_size=1024, device="cuda"):
    dataset = torch.utils.data.TensorDataset(x_train.float())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_sample = x_train.shape[0]
    output = torch.zeros((num_sample, n_classes))

    with torch.no_grad():
        for batch_idx, x_batch in enumerate(data_loader):
            x = x_batch[0].to(device)
            pred = model(x)
            output[batch_idx * batch_size: batch_idx * batch_size + x.size(0)] = pred.cpu()

    return output


def get_precision_recall(pred, target):
    # Calculate TP, FP, FN
    TP = torch.sum((pred == 1) & (target == 1))
    FP = torch.sum((pred == 1) & (target == 0))
    FN = torch.sum((pred == 0) & (target == 1))

    # Calculate Precision and Recall
    precision = TP.float() / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP.float() / (TP + FN) if (TP + FN) > 0 else 0

    return precision.item(), recall.item()


def get_consensus_rate(teacher_prev, teacher_curr, ground_truth=None):
    consensus_tt = (teacher_prev.argmax(dim=1) == teacher_curr.argmax(dim=1))
    rate_top1_consensus = (consensus_tt).sum() / consensus_tt.shape[0]
    # print(f'teacher-teacher top1 consensus rate: {rate_top1_consensus:.2f}')

    consensus_top2 = get_conservative_estimate(teacher_curr, teacher_prev, 2)
    rate_top2_consensus = (consensus_top2).sum() / consensus_top2.shape[0]
    # print(f'teacher-teacher top2 consensus rate: {rate_top2_consensus:.2f}')

    consensus_top3 = get_conservative_estimate(teacher_curr, teacher_prev, 3)
    rate_top3_consensus = (consensus_top3).sum() / consensus_top3.shape[0]
    # rate_top3_consensus = torch.tensor(0.0)
    print(f'teacher-teacher top3 consensus rate: {rate_top3_consensus:.2f}')

    if ground_truth is not None:
        consensus_prev = (teacher_prev.argmax(dim=1) == ground_truth)
        rate_prev = (consensus_prev).sum() / consensus_prev.shape[0]
        print(f'previous teacher accuracy: {rate_prev:.2f}')

        consensus_curr = (teacher_curr.argmax(dim=1) == ground_truth)
        rate_curr = (consensus_curr).sum() / consensus_curr.shape[0]
        print(f'current teacher accuracy: {rate_curr:.2f}')

        precision_prev, recall_prev = get_precision_recall(consensus_tt, consensus_prev)
        print(f'consensus for previous teacher: precision = {precision_prev:.2f}, recall = {recall_prev:.2f}')
        precision_curr, recall_curr = get_precision_recall(consensus_tt, consensus_curr)
        print(f'consensus for current teacher: precision = {precision_curr:.2f}, recall = {recall_curr:.2f}')

        # confidence consistent
        p_prev, y_prev = teacher_prev.max(dim=1)
        p_curr, y_curr = teacher_curr.max(dim=1)
        consistent_tt = consensus_tt & (p_curr >= p_prev)
        rate_tt_consistent = (consistent_tt).sum() / consistent_tt.shape[0]
        print(f'teacher-teacher consistent rate: {rate_tt_consistent:.2f}')

        precision_prev, recall_prev = get_precision_recall(consistent_tt, consensus_prev)
        print(f'consistent for previous teacher: precision = {precision_prev:.2f}, recall = {recall_prev:.2f}')
        precision_curr, recall_curr = get_precision_recall(consistent_tt, consensus_curr)
        print(f'consistent for current teacher: precision = {precision_curr:.2f}, recall = {recall_curr:.2f}')

    return rate_top1_consensus.item(), rate_top2_consensus.item(), rate_top3_consensus.item()


def probe(d, n_classes, n_hidden=0, device="cuda"):
    if n_hidden == 0:
        model = nn.Linear(d, n_classes).to(device)
    elif n_hidden == 1:
        h = int(d/2)
        model = nn.Sequential(
                    nn.Linear(d, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, n_classes)
                ).to(device)
    elif n_hidden == 2:
        h = int(d/2)
        model = nn.Sequential(
                    nn.Linear(d, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, n_classes)
                ).to(device)
    else:
        raise NotImplementedError
    return model


def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    n_epochs=10,
    weight_decay=0.0,
    lr=1.0e-3,
    batch_size=100,
    n_classes=1000,
    model=None,
    device="cuda",
):
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0)

    d = x_train.shape[1]
    if model is None:
        model = probe(d, n_classes)
        print('Initialize model')
    else:
        print('Warm-start model')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["train_all"] = []

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
        results["train_all"].append(correct / total)

        for key, (x_test, y_test) in eval_datasets.items():
            x_test = x_test.float().to(device)
            pred = torch.argmax(model(x_test), axis=-1).detach()
            acc = (pred == y_test).float().mean().item()
            results[f"{key}_all"].append(acc)

        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}, Test Acc {results['test_all'][-1]:.3f}")

    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    return results


def main(cfg_path: str):
    ### load cfg
    cfg = YAML().load(open(cfg_path))

    ### saving and logging
    cfg["run_name"] = (run_name := datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    cache_path = os.path.join(cfg["save_path"], slugify(cfg['data']['name']), "cache")
    os.makedirs(cache_path, exist_ok=True)
    results_dir = os.path.join(cfg["save_path"], slugify(cfg['data']['name']), "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving to {results_dir}")

    ### reproducibility
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    ### get data
    print("Loading the data...")
    dsets, dls = get_data(cfg=cfg["data"])
    n_classes = dsets[tuple(dsets.keys())[0]].n_classes if hasattr(dsets[tuple(dsets.keys())[0]], "n_classes") else len(dsets[tuple(dsets.keys())[0]].classes)

    ### run for different data setups
    for ds_i, data_setup_cfg in enumerate(cfg["csl"]["setups"]):
        assert set(data_setup_cfg.keys()) == {"train", "test"} or set(data_setup_cfg.keys()) == {"split"}, \
            "Data setup must have either 'train' and 'test' or 'split' keys."

        label_layers = defaultdict(list)
        for tier in data_setup_cfg:
            all_teachers_label_paths = data_setup_cfg[tier]["teacher_label_paths"]
            student_emb_path = data_setup_cfg[tier]["student_emb_path"]

            ### collect all teacher labels
            label_teachers, acc_teachers = [], []
            for teacher_labels_path in all_teachers_label_paths:
                print(f"Loading teacher labels at layer {len(label_layers[tier])} from {teacher_labels_path}...")
                if teacher_labels_path is None:
                    ### signal that a new layer is starting
                    print(f"Accuracies: {[round(acc, 4) for acc in acc_teachers]}")
                    print("Starting a new layer...")
                    label_layers[tier].append(label_teachers)
                    label_teachers = []
                    acc_teachers = []
                    continue

                ### load teacher labels
                ckpt = torch.load(teacher_labels_path, pickle_module=dill, map_location="cpu")
                gt_labels, weak_labels, weak_acc = ckpt["gt_labels"].to(cfg["device"]), ckpt["teacher_labels"].to(cfg["device"]), ckpt["teacher_acc"]

                ### convert to hard labels
                if not cfg["csl"]["soft_teacher"]:
                    weak_labels = nn.functional.one_hot(torch.argmax(weak_labels, dim=1), n_classes=n_classes).float()
                    print('Convert teacher outputs to hard class labels')

                ### use only a fraction of the data
                if data_setup_cfg[tier]["frac_max"] is not None:
                    n_max = int(data_setup_cfg[tier]["frac_max"] * len(weak_labels))
                    weak_labels = weak_labels[:n_max]
                    gt_labels = gt_labels[:n_max]

                assert weak_labels.ndim == 2 or weak_labels.shape[1] == 1, "Loaded more teachers than allowed (1)."
                weak_labels = weak_labels if weak_labels.ndim == 2 else weak_labels.squeeze(1)
                label_teachers.append(weak_labels)
                try:
                    acc_teachers.append(weak_acc[0].item())
                except:
                    acc_teachers.append(weak_acc.item())

            ### append last layer
            print(f"Accuracies: {[round(acc, 4) for acc in acc_teachers]}")
            label_layers[tier].append(label_teachers)
            print(f"Number of layers: {len(label_layers[tier])}")

            ### get embeddings of the student model
            print(f"Loading student model embeddings from {student_emb_path}...")
            cached_embs = torch.load(student_emb_path, pickle_module=dill, map_location="cpu")
            embeddings, student_gt_labels = cached_embs["embeddings"].to(cfg["device"]), cached_embs["gt_labels"].to(cfg["device"])
            if data_setup_cfg[tier]["frac_max"] is not None:
                n_max = int(data_setup_cfg[tier]["frac_max"] * len(embeddings))
                embeddings, student_gt_labels = embeddings[:n_max], student_gt_labels[:n_max]
            if tier != "test":
                assert torch.all(gt_labels.to(cfg["device"]) == student_gt_labels), "Ground truth labels must be the same for teachers and student."

            ### prepare train and test datasets
            if tier == "split":
                ### prepare train and test for w2s-csl training by splitting the loaded labels and embeddings
                order = np.arange(len(embeddings))
                rng.shuffle(order)
                x = embeddings[order]
                y = gt_labels[order]

                ### split
                assert len(cfg["w2s"]["train_val_test_split"]) == 3, "Train, val, test split must be of length 3."
                assert sum(cfg["w2s"]["train_val_test_split"]) == 1.0, "Train, val, test split must sum to 1."
                n_train, n_val = int(cfg["w2s"]["train_val_test_split"][0] * len(order)), int(cfg["w2s"]["train_val_test_split"][1] * len(order))
                x_train, x_val, x_test = x[:n_train], x[n_train:n_train+n_val], x[n_train+n_val:]
                y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
                eval_datasets = {"val": (x_val, y_val), "test": (x_test, y_test)}
                print(f"Number of samples in train, val, test: {len(x_train)}, {len(x_val)}, {len(x_test)}")
            elif tier == "train":
                ### use the entire dataset for training
                order = np.arange(len(embeddings)) # shuffle the dataset
                rng.shuffle(order)
                x_train, y_train = embeddings[order], gt_labels[order]
            elif tier == "test":
                ### use the entire dataset for testing
                x_test, y_test = embeddings, student_gt_labels
                eval_datasets = {"test": (x_test, y_test)}
            else:
                raise ValueError

        print(f"Number of training samples: {len(x_train)}.")
        print(f"Number of testing samples: {len(x_test)}.")
        if "train" in data_setup_cfg:
            assert len(label_layers["train"]) == len(label_layers["test"]), "Number of layers must be the same for train and test."
            x = torch.cat([x_train, x_test], dim=0)

        ### main loop - from top to bottom of the hierarchy
        result_teacher, result_teacher_test, result_student, result_student_val, result_rate, result_precision, result_recall = [], [], [], [], [], [], []
        logit_prev, logit_prev_all = None, None
        for layer_idx, label_teachers in enumerate(label_layers["train" if "train" in data_setup_cfg else "split"]):
            num_teacher = len(label_teachers)

            print("\n" + "-" * 50)
            print(f"Layer {layer_idx}. Number of teachers: {num_teacher}")
            print("-" * 50)

            ### teacher assignment
            if num_teacher > 1:
                yw = torch.stack(label_teachers, dim=1)[order] # [N, num_teacher, n_classes]
                if "test" in data_setup_cfg:
                    yw = torch.cat([yw, torch.stack(label_layers["test"][layer_idx], dim=1)], dim=0) # [N := N_train + N_test, num_teacher, n_classes]
                    n_train = len(x_train)

                if cfg["csl"]["teacher_assignment"] == "ensemble":
                    ### do an ensemble of all teachers
                    yw = yw.mean(dim=1)
                    yw /= yw.sum(dim=1, keepdim=True)
                elif cfg["csl"]["teacher_assignment"] == "oracle":
                    ### use the teacher that has the lowest loss on the training set (per-sample)
                    losses = []
                    for teacher_idx in range(yw.shape[1]):
                        losses.append(nn.functional.nll_loss(
                            yw[:, teacher_idx].log(),
                            y,
                            reduction='none',
                        ))
                    losses = torch.stack(losses, dim=1) # [N, num_teacher]
                    teacher_assignment = torch.argmin(losses, dim=1)
                    yw = yw[torch.arange(yw.shape[0]), teacher_assignment]
                elif cfg["csl"]["teacher_assignment"] == "student_teacher_agreement":
                    ### use the soft labels of the teacher that agrees with the student the most (per-sample)
                    disagreements = []
                    for teacher_idx in range(yw.shape[1]):
                        disagreements.append(nn.functional.kl_div(
                            yw[:, teacher_idx].log(),
                            nn.functional.log_softmax(logit_prev_all, dim=1),
                            reduction='none',
                            log_target=True,
                        ).sum(dim=1))
                    disagreements = torch.stack(disagreements, dim=1) # [N, num_teacher]
                    teacher_assignment = torch.argmin(disagreements, dim=1)
                    yw = yw[torch.arange(yw.shape[0]), teacher_assignment]
                else:
                    raise NotImplementedError

                # print("Teacher accuracy on the test set:")
                # print([tacc.item() for tacc in ((torch.stack(label_teachers, dim=1)[order][n_train:].argmax(-1) == y_test.unsqueeze(-1)).sum(0) / len(y_test))])
            else:
                yw = label_teachers[0][order]
                if "test" in data_setup_cfg:
                    yw = torch.cat([yw, label_layers["test"][layer_idx][0]], dim=0) # [N := N_train + N_test, n_classes]
                    n_train = len(x_train)
                    n_val = 0

            ### split the weak labels
            yw_train = yw[:n_train]
            yw_val = yw[n_train:n_train+n_val]
            yw_test = yw[n_train+n_val:]

            ### teacher(s) accuracy
            acc_teacher = ((yw_train.argmax(dim=1) == y_train).sum() / n_train).item()
            acc_teacher_test = ((yw_test.argmax(dim=1) == y_test).sum() / len(y_test)).item()
            result_teacher.append(acc_teacher)
            result_teacher_test.append(acc_teacher_test)
            print(f"{num_teacher} teachers' collective accuracy:  train = {acc_teacher:.4f}, test = {acc_teacher_test:.4f}")

            ### sample filtering
            if num_teacher > 1 and cfg["csl"]["denoise_criterion"] != 'all':
                rate_top1, rate_top2, rate_top3 = get_consensus_rate(torch.nn.functional.softmax(logit_prev, dim=1), yw_train)
                if cfg["csl"]["denoise_criterion"] == 'top1':
                    rate_keep = rate_top1
                elif cfg["csl"]["denoise_criterion"] == 'top2':
                    rate_keep = rate_top2
                elif cfg["csl"]["denoise_criterion"] == 'top3':
                    rate_keep = rate_top3
                elif cfg["csl"]["denoise_criterion"] == 'oracle':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                rate_keep = 1.0

            if rate_keep < 1.0:
                rank_train = get_student_rank(logit_prev, yw_train.argmax(dim=1))
                keep_train = rank_train < (n_train * rate_keep)
            else:
                keep_train = torch.ones(n_train, dtype=torch.bool, device=y_train.device)

            ### precision and recall
            correct_train = (yw_train.argmax(dim=1) == y_train)
            precision, recall = get_precision_recall(keep_train, correct_train)
            print(f"Selecting {rate_keep*100:.1f}% samples:\n  precision = {precision:.3f}\n  recall = {recall:.3f}")
            result_rate.append(rate_keep)
            result_precision.append(precision)
            result_recall.append(recall)

            ### train
            print("-" * 30)
            print(f"Training logreg on the selected weak labels...")
            epochs_conserve = int(cfg["w2s"]["n_epochs"] / rate_keep) # keep the same number of iterations
            x_conserve, yw_conserve = x_train[keep_train], yw_train[keep_train]
            print(f"  Number of samples: {len(x_conserve)} (out of {len(x_train)})")
            print(f"  Number of epochs: {epochs_conserve}")
            if cfg["csl"]["reinit_student_head_at_each_layer"] or layer_idx == 0:
                model = probe(d=x_train.shape[1], n_classes=n_classes, device=cfg["device"])
            results_weak = train_logreg(x_conserve, yw_conserve, eval_datasets, model=model,
                n_epochs=epochs_conserve, lr=cfg["w2s"]["lr"], batch_size=cfg["w2s"]["batch_size"], device=cfg["device"])
            print(f"Accuracies per epoch: {[round(acc, 4) for acc in results_weak['test_all']]}")
            print(f"Final accuracy: {results_weak['test']:.4f}")
            result_student.append(results_weak["test"])
            result_student_val.append(results_weak.get("val", None))

            ### update and go one layer down
            logit_prev = get_output(model, x_train, n_classes=n_classes, device=cfg["device"]).to(cfg["device"])
            logit_prev_all = get_output(model, x, n_classes=n_classes, device=cfg["device"]).to(cfg["device"])

        print(f"\n--- Accuracies by layer ---")
        print(f"Teacher on training set: {[round(acc, 4) for acc in result_teacher]}")
        print(f"Teacher on test set: {[round(acc, 4) for acc in result_teacher_test]}")
        print(f"Student on test set: {[round(acc, 4) for acc in result_student]}")
        print(f"Student on validation set: {[round(acc, 4) for acc in result_student_val]}")
        print(f"--- Rates ---")
        print(f"Rate: {[round(rate, 4) for rate in result_rate]}")
        print(f"--- Precision ---")
        print(f"Precision: {[round(precision, 4) for precision in result_precision]}")
        print(f"--- Recall ---")
        print(f"Recall: {[round(recall, 4) for recall in result_recall]}")
        print(f"--- Done ---")

        ### save results
        save_to = os.path.join(results_dir, f"{run_name}_{ds_i}.pt")
        torch.save({
            "cfg": cfg,
            'rate': result_rate,
            'precision': result_precision,
            'recall': result_recall,
            'teacher': result_teacher,
            'teacher_test': result_teacher_test,
            'student': result_student,
            'student_val': result_student_val,
        }, save_to, pickle_module=dill)
        print(f"Results saved to {save_to}")



if __name__ == "__main__":
    fire.Fire(main)