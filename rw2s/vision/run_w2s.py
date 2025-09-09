import os
import sys
from copy import deepcopy
import fire
import numpy as np
from collections import defaultdict
from datetime import datetime
import tqdm
import dill
from loguru import logger
from ruamel.yaml import YAML
import torch
import wandb
import lovely_tensors as lt
from functools import partial

from rw2s.utils import preload, get_cache_paths, seed_all, slugify
from rw2s.losses import LOSS_DICT
from rw2s.train import train, train_head
from rw2s.vision.data import get_data, filter_dl_by_metadata
from rw2s.vision.models import get_model, freeze_backbone, LinearProbeClassifier
from rw2s.vision.ensemble import Ensemble, EnsembleParticipant, EnsembleDisagreementSchedule

np.set_printoptions(threshold=sys.maxsize)
lt.monkey_patch()
wandb.login()



def setup_logging(cfg):
    ### saving and logging
    cfg["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg["ensemble_model_name"] = '_'.join([model_cfg["model_name"] for model_cfg in cfg["ensemble"]["weak_models"]])
    cfg["run_name_long"] = f"{cfg['run_name']}__{slugify(cfg['data']['name'])}__{cfg['ensemble_model_name']}__{'_'.join([sm['model_name'] for sm in cfg['w2s']['student_models']])}"
    cfg["run_name_long"] = cfg["run_name_long"].replace("/", "_").replace(" ", "_")
    cfg["cache_path"] = os.path.join(cfg["save_path"], slugify(cfg['data']['name']), "cache")
    cfg["results_dir"] = os.path.join(cfg["save_path"], slugify(cfg['data']['name']), "results")
    os.makedirs(cfg["cache_path"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    ### local logging
    logger.remove()
    logger.add(sys.stdout, format="{message}")

    ### wandb
    wdb_run = None
    if cfg["track_in_wandb"]:
        wdb_run = wandb.init(project="rw2s", name=cfg["run_name_long"], id=cfg["run_name"], config=cfg, **(cfg.get("wandb_kwargs", None) or {}))

    return cfg, wdb_run, logger


def get_weak_model(cfg, model_cfg, logger, wdb_run, n_classes, model_idx=None):
    weak_model = get_model(name=model_cfg["model_name"], device=cfg["device"],
        pretrained=model_cfg["pretrained"], replace_last_layer_with_n_classes=n_classes)

    ### load ckpt
    if model_cfg["load_ckpt"]:
        logger.info(f"Loading the weak model's checkpoint from {model_cfg['load_ckpt']}...")
        weak_model.load_state_dict(torch.load(model_cfg["load_ckpt"], pickle_module=dill, map_location=torch.device(cfg["device"]))["state_dict"])

    ### freeze backbone
    if model_cfg["freeze_backbone"]:
        weak_model = freeze_backbone(weak_model, model_name=model_cfg["model_name"])

    if wdb_run: wdb_run.watch(weak_model, idx=model_idx)

    ### train
    if model_cfg["train"]:
        ### prepare its training data
        dsets, dls = get_data(cfg=cfg["data"])
        if cfg["data"].get("subsplit_train", None) is None:
            train_dl, val_dl = dls["train"], dls["id_val"]
        else:
            train_dl, val_dl = dls["train_split0"], dls["train_split1"] # subsplit train into train and val
        if model_cfg["train_cfg"]["domain_start_end_idxs"] is not None:
            train_dl, domain_start_end_val_range = filter_dl_by_metadata(dataloader=train_dl, meta_key=model_cfg["train_cfg"]["domain_group"], meta_start_end_idxs=model_cfg["train_cfg"]["domain_start_end_idxs"])
            val_dl, _ = filter_dl_by_metadata(dataloader=val_dl, meta_key=model_cfg["train_cfg"]["domain_group"], meta_start_end_val_range=model_cfg["train_cfg"]["domain_start_end_idxs"])

        ### train
        logger.info(f"Training the weak model ({len(train_dl.dataset)} training samples, {len(val_dl.dataset) if val_dl else 0} validation samples)...")
        save_path = os.path.join(cfg["save_path"], slugify(cfg['data']['name']), "models",
            f"{cfg['run_name']}__weak__{model_cfg['model_name']}{'__'+str(model_idx) if model_idx is not None else ''}.pt")
        weak_model = train(
            model=weak_model,
            train_dl=train_dl,
            loss_fn=LOSS_DICT[model_cfg["train_cfg"]["loss_fn_name"]](**(model_cfg["train_cfg"]["loss_fn_kwargs"] or dict())),
            val_dl=val_dl,
            val_loss_fn=LOSS_DICT["xent"](),
            n_epochs=model_cfg["train_cfg"]["n_epochs"],
            optimizer_name=model_cfg["train_cfg"]["optimizer_name"],
            optimizer_kwargs=model_cfg["train_cfg"]["optimizer_kwargs"],
            scheduler_name=model_cfg["train_cfg"]["scheduler_name"],
            scheduler_mul_factor=model_cfg["train_cfg"]["scheduler_mul_factor"],
            early_stopping_patience=model_cfg["train_cfg"]["early_stopping_patience"],
            load_best_model=model_cfg["train_cfg"]["load_best_model"],
            device=cfg["device"],
            logger=logger,
            wdb_run=wdb_run,
            ckpt_path=save_path.replace(".pt", "_latest.pt"),
        )

        ### save
        logger.info(f"Saving the weak model to {save_path}...")
        torch.save({
            "cfg": cfg,
            "model_cfg": model_cfg,
            "state_dict": weak_model.state_dict(),
        }, save_path, pickle_module=dill)

    return weak_model


def get_ensemble(cfg, logger, wdb_run, dls, n_classes):
    ### load weak models
    weak_models = []
    for m_i, model_cfg in enumerate(cfg["ensemble"]["weak_models"]):
        logger.info(f"Initializing weak {model_cfg['model_name']} (pretrained={model_cfg['pretrained']})...")
        weak_model = get_weak_model(cfg=cfg, model_cfg=model_cfg, logger=logger, wdb_run=wdb_run, n_classes=n_classes, model_idx=m_i)
        weak_models.append(EnsembleParticipant(model=weak_model, cfg=model_cfg))

    ### init ensemble
    logger.info(f"Initializing ensemble of {len(weak_models)} weak models...")
    weak_ensemble = Ensemble(
        models=weak_models,
        ensemble_weights_init=cfg["ensemble"]["ensemble_weights_init"],
        reset_ensemble_weights_at_epoch_end_to_uniform=cfg["ensemble"]["reset_ensemble_weights_at_epoch_end_to_uniform"],
        ensemble_weights_per_n_samples=1,
        device=cfg["device"],
        dataloader=dls["id_val"] if cfg["ensemble"]["ensemble_weights_init"] in ("entropy","kl_disagreement") else None,
    )
    if wdb_run: wdb_run.watch(weak_ensemble)

    return weak_ensemble


def prepare_w2s_callbacks(teacher_model, cfg, logger, bootstrap_step, combine_logits=True):
    before_optim_run_callback_weak, before_batch_callback_weak, after_batch_callback_weak = None, None, None
    if cfg["ensemble"]["ensemble_weights_opter_cfg"] \
        and (bootstrap_step == 0 or cfg["w2s"]["add_students_to_ensemble"]):
        logger.info(f"Ensemble weights opter cfg: {cfg['ensemble']['ensemble_weights_opter_cfg']}")
        logger.info(f"Ensemble weights per sample: {cfg['ensemble'].get('ensemble_weights_per_sample', None)}")
        logger.info(f"Ensemble weights freeze first n epochs: {(ensemble_weights_optim_skip_first_epochs := (int((cfg['ensemble']['ensemble_weights_freeze_first_frac_of_epochs'] or 0) * cfg['w2s']['n_epochs'])))}")
        before_optim_run_callback_weak = lambda yw, sample_idxs, **kwargs: teacher_model.init_ens_ws_optim_run(
            ensemble_weights_opter_cfg=cfg["ensemble"]["ensemble_weights_opter_cfg"],
            yw=yw,
            force_ens_ws_n_samples=len(sample_idxs) if cfg["ensemble"]["ensemble_weights_per_sample"] else None,
            ensemble_weights_optim_skip_first_epochs=ensemble_weights_optim_skip_first_epochs,
        )
        sample_ws_schedule = EnsembleDisagreementSchedule(apply_first_n_epochs=cfg["w2s"]["ignore_samples_with_disagreement_first_n_epochs"] or 0)
        before_batch_callback_weak = lambda x, y, pred, sample_idxs, sample_ws, epoch, **kwargs: (
            x,
            *(teacher_model.combine_logits(y_hats=y, pred=pred, x=x, sample_idxs=sample_idxs, treat_as_probas=True) if combine_logits else (y, pred)),
            sample_ws_schedule(sample_ws=sample_ws, yw=y, epoch=epoch),
        )
        after_batch_callback_weak = partial(teacher_model.ensemble_weights_update_callback, call_backward=False,
            keep_history=cfg["ensemble"]["save_ensemble_weights"] == "all" or cfg["ensemble"]["save_ensemble_weights"].startswith("every_"))

    return before_optim_run_callback_weak, before_batch_callback_weak, after_batch_callback_weak


def run_w2s(cfg, logger, dls, n_classes, results, teacher_model, student_model, teacher_model_name, student_model_name, bootstrap_step):
    ### train/eval w2s on id_val data
    if cfg["w2s"]["eval_on_id_val_data"]:
        cached_labels_path, cached_embs_path = get_cache_paths(
            cfg=cfg, student_model_name=student_model_name, teacher_model_name=teacher_model_name, cache_dir_suffix="__val")

        ### prepare callbacks for ensemble weights optimization
        before_optim_run_callback_weak, before_batch_callback_weak, after_batch_callback_weak = prepare_w2s_callbacks(
            teacher_model=teacher_model, cfg=cfg, logger=logger, bootstrap_step=bootstrap_step, combine_logits=cfg["w2s"]["teacher_labels_loss_fn_name"] != "edl")

        ### run
        seed_all(cfg["seed"])
        results_id_val, student_model_probe, student_model_probe_data = train_head(
            teacher_model=teacher_model,
            student_model=student_model,
            dataloader=dls[cfg["w2s"]["id_val_data_key"]],
            cfg=cfg,
            cached_labels_path=cached_labels_path,
            cached_embs_path=cached_embs_path,
            logger=logger,
            results=defaultdict(list), # don't overwrite the results on test data
            rng=np.random.default_rng(cfg["seed"]),
            n_classes=n_classes,
            return_data=True,
            before_optim_run_callback_weak=before_optim_run_callback_weak,
            before_batch_callback_weak=before_batch_callback_weak,
            after_batch_callback_weak=after_batch_callback_weak,
        )

        ### save results
        results["w2s_id_val"].append(results_id_val)
        if cfg["ensemble"]["save_ensemble_weights"] == "all":
            results["w2s_id_val"][-1]["ensemble_weights"].append(teacher_model.history["ensemble_weights"])
        elif cfg["ensemble"]["save_ensemble_weights"] == "last":
            results["w2s_id_val"][-1]["ensemble_weights"].append(teacher_model.ens_ws.detach().cpu().clone())
        elif cfg["ensemble"]["save_ensemble_weights"].startswith("every_"):
            save_every_nth_step = int(cfg["ensemble"]["save_ensemble_weights"].split("_")[-1])
            for i in range(0, len(teacher_model.history["ensemble_weights"]), save_every_nth_step):
                results["w2s_id_val"][-1]["ensemble_weights"].append(teacher_model.history["ensemble_weights"][i])

    ### train/eval w2s on test data (and test on id val data if eval_on_id_val_data)
    cached_labels_path, cached_embs_path = get_cache_paths(
        cfg=cfg, student_model_name=student_model_name, teacher_model_name=teacher_model_name)

    ### prepare callbacks for ensemble weights optimization
    before_optim_run_callback_weak, before_batch_callback_weak, after_batch_callback_weak = prepare_w2s_callbacks(
        teacher_model=teacher_model, cfg=cfg, logger=logger, bootstrap_step=bootstrap_step, combine_logits=cfg["w2s"]["teacher_labels_loss_fn_name"] != "edl")

    ### run
    seed_all(cfg["seed"])
    results, student_model_probe = train_head(
        teacher_model=teacher_model,
        student_model=student_model,
        dataloader=dls[cfg["w2s"]["test_data_key"]],
        cfg=cfg,
        cached_labels_path=cached_labels_path,
        cached_embs_path=cached_embs_path,
        logger=logger,
        results=results,
        rng=np.random.default_rng(cfg["seed"]),
        n_classes=n_classes,
        return_data=False,
        additional_eval_data=None if not cfg["w2s"]["eval_on_id_val_data"] else {
            "id_val_all": (student_model_probe_data["x"], student_model_probe_data["y"]),
            "id_val_all_weak": (student_model_probe_data["x"], torch.argmax(student_model_probe_data["yw"], dim=1)),
            "id_val_val": (student_model_probe_data["x_val"], student_model_probe_data["y_val"]),
            "id_val_val_weak": (student_model_probe_data["x_val"], student_model_probe_data["yw_val"]),
            "id_val_test": (student_model_probe_data["x_test"], student_model_probe_data["y_test"]),
            "id_val_test_weak": (student_model_probe_data["x_test"], student_model_probe_data["yw_test"]),
        },
        before_optim_run_callback_weak=before_optim_run_callback_weak,
        before_batch_callback_weak=before_batch_callback_weak,
        after_batch_callback_weak=after_batch_callback_weak,
    )

    ### save ensemble weights
    if cfg["ensemble"]["save_ensemble_weights"] == "all":
        results["ensemble_weights"].append(teacher_model.history["ensemble_weights"])
    elif cfg["ensemble"]["save_ensemble_weights"] == "last":
        results["ensemble_weights"].append(teacher_model.ens_ws.detach().cpu().clone())
    elif cfg["ensemble"]["save_ensemble_weights"].startswith("every_"):
        save_every_nth_step = int(cfg["ensemble"]["save_ensemble_weights"].split("_")[-1])
        for i in range(0, len(teacher_model.history["ensemble_weights"]), save_every_nth_step):
            results["ensemble_weights"].append(teacher_model.history["ensemble_weights"][i])
    
    return results, student_model, student_model_probe


def main(cfg_path: str):
    ### load cfg
    cfg = YAML().load(open(cfg_path))

    ### saving and logging
    cfg, wdb_run, logger = setup_logging(cfg)

    ### reproducibility
    seed_all(cfg["seed"])

    ### get data
    logger.info("Loading the data...")
    dsets, dls = get_data(cfg=cfg["data"])
    n_classes = dsets[tuple(dsets.keys())[0]].n_classes if hasattr(dsets[tuple(dsets.keys())[0]], "n_classes") else len(dsets[tuple(dsets.keys())[0]].classes)

    ### load weak ensemble
    weak_ensemble = get_ensemble(cfg=cfg, logger=logger, wdb_run=wdb_run, dls=dls, n_classes=n_classes)

    ### run boostrapping (1-step bootstrapping == normal w2s)
    teacher_model, teacher_model_name = weak_ensemble, cfg["ensemble_model_name"]
    results = defaultdict(list)
    for bootstrap_step, student_model_cfg in enumerate(cfg["w2s"]["student_models"]):
        ### load student model
        logger.info(f"Loading the student model ({'' if student_model_cfg['pretrained'] else 'not '}pretrained {student_model_cfg['model_name']})...")
        student_model = get_model(name=student_model_cfg["model_name"], device=cfg["device"], pretrained=student_model_cfg["pretrained"])
        if student_model_cfg["load_probe_ckpt"]:
            student_model_probe = torch.load(student_model_cfg["load_probe_ckpt"], pickle_module=dill, map_location=torch.device(cfg["device"]))["state_dict"]
            student_model = LinearProbeClassifier(backbone=student_model, classifier=student_model_probe)
        if wdb_run: wdb_run.watch(student_model)

        ### w2s
        results, student_model, student_model_probe = run_w2s(cfg=cfg, logger=logger, dls=dls, n_classes=n_classes, results=results,
            teacher_model=teacher_model, student_model=student_model, teacher_model_name=teacher_model_name, student_model_name=student_model_cfg["model_name"], bootstrap_step=bootstrap_step)

        ### logging
        # teacher labels
        logger.info("\n" + "-" * 10)
        logger.info("Training logreg on teacher (pseudo) labels...")
        logger.info(f"Final accuracy: {results['results_teacher_to_student'][-1]['test']:.3f}")
        logger.info(f"Final supervisor-student agreement: {results['results_teacher_to_student'][-1]['test_weak']:.3f}")
        logger.info(f"Accuracy by epoch: {[round(acc.item(), 4) for acc in results['results_teacher_to_student'][-1]['test_all']]}")
        logger.info(f"Supervisor-student agreement by epoch: {[round(acc.item(), 4) for acc in results['results_teacher_to_student'][-1]['test_weak_all']]}")
        logger.info("-" * 10)

        # gt labels
        logger.info("\n" + "-" * 10)
        logger.info("Training logreg on ground truth labels...")
        logger.info(f"Final accuracy: {results['results_gt'][-1]['test']:.3f}")
        logger.info(f"Accuracy by epoch: {[round(acc.item(), 4) for acc in results['results_gt'][-1]['test_all']]}")
        logger.info("-" * 10)

        # final
        logger.info("\n" + "=" * 10 + f" {teacher_model_name} → {student_model_cfg['model_name']} " + "=" * 10)
        logger.info(f"Teacher label accuracy:\n  all data:\t\t{results['teacher_acc'][-1]:.3f}\n  val data:\t\t{results['teacher_acc_val'][-1]:.3f}\n  test data:\t\t{results['teacher_acc_test'][-1]:.3f}")
        if cfg["w2s"]["eval_on_id_val_data"]: logger.info(f"  ID val data (all):\t{results['w2s_id_val'][-1]['teacher_acc'][-1]:.3f}\n  ID val-val data:\t{results['w2s_id_val'][-1]['teacher_acc_val'][-1]:.3f}\n  ID val-test data:\t{results['w2s_id_val'][-1]['teacher_acc_test'][-1]:.3f}")
        logger.info(f"Teacher → Student accuracy:\n  val data:\t\t{results['results_teacher_to_student'][-1]['val']:.3f}\n  test data:\t\t{results['results_teacher_to_student'][-1]['test']:.3f}")
        if cfg["w2s"]["eval_on_id_val_data"]: logger.info(f"  ID val-val data:\t{results['w2s_id_val'][-1]['results_teacher_to_student'][-1]['val']:.3f}\n  ID val-test data:\t{results['w2s_id_val'][-1]['results_teacher_to_student'][-1]['test']:.3f}")
        logger.info(f"GT → Student accuracy:\n  val data:\t\t{results['results_gt'][-1]['val']:.3f}\n  test data:\t\t{results['results_gt'][-1]['test']:.3f}")
        if cfg["w2s"]["eval_on_id_val_data"]:
            logger.info(f"  ID val data:\t\t{results['w2s_id_val'][-1]['results_gt'][-1]['test']:.3f}")
            logger.info(f"Teacher → Student accuracy (Double OOD):\n  ID val data (all):\t{results['results_teacher_to_student'][-1]['id_val_all']:.3f}\n  ID val-val data:\t{results['results_teacher_to_student'][-1]['id_val_val']:.3f}\n  ID val-test data:\t{results['results_teacher_to_student'][-1]['id_val_test']:.3f}")
            logger.info(f"GT → Student accuracy (Double OOD):\n  ID val data (all):\t{results['results_gt'][-1]['id_val_all']:.3f}\n  ID val-val data:\t{results['results_gt'][-1]['id_val_val']:.3f}\n  ID val-test data:\t{results['results_gt'][-1]['id_val_test']:.3f}")
        logger.info("=" * (20 + len(f" {teacher_model_name} → {student_model_cfg['model_name']} ")))

        ### update teacher
        teacher_model = EnsembleParticipant(model=LinearProbeClassifier(backbone=student_model, classifier=student_model_probe), cfg=student_model_cfg)
        if cfg["w2s"]["add_students_to_ensemble"]:
            ### append student model to the current ensemble
            weak_ensemble.add_member(model=teacher_model)
            teacher_model = weak_ensemble
            teacher_model_name = f"{teacher_model_name}__{student_model_cfg['model_name']}"
        else:
            ### student is now the new (single) teacher (boostrapping)
            teacher_model = Ensemble(
                models=[teacher_model],
                ensemble_weights_init=cfg["ensemble"]["ensemble_weights_init"],
                reset_ensemble_weights_at_epoch_end_to_uniform=cfg["ensemble"]["reset_ensemble_weights_at_epoch_end_to_uniform"],
                ensemble_weights_per_n_samples=1,
                device=cfg["device"],
                dataloader=dls["id_val"] if cfg["ensemble"]["ensemble_weights_init"] in ("entropy","kl_disagreement") else None,
            )
            teacher_model_name = student_model_cfg["model_name"]

    ### save results
    results_path = os.path.join(cfg['results_dir'], f"{cfg['run_name_long']}.pt")
    results["cfg"] = cfg
    torch.save(dict(results), results_path, pickle_module=dill)
    logger.info(f"Results saved to {results_path}")

    return dict(results)


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda x: None) # run and suppress printing of the return value
