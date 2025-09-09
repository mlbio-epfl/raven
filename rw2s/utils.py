import os
import random
import numpy as np
import torch
import unicodedata
import re
import tqdm
import dill


@torch.no_grad()
def preload(
    model,
    loader,
    device,
    store_embs=True,
    store_inps=False,
    store_domains=False,
    chunking_dir=None,
    chunk_size=20,
):
    if chunking_dir is not None:
        os.makedirs(chunking_dir, exist_ok=True)
    all_x, all_embeddings, all_y, all_probs, all_domains = [], [], [], [], []
    chunk_files_created = []
    n_batches = len(loader)
    for b_idx, b in tqdm.tqdm(enumerate(loader), total=n_batches):
        x, y = b[:2]
        embeddings = model(x.to(device))
        if len(embeddings) == 2 and type(embeddings) in (list, tuple):
            embeddings, logits = embeddings
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
            all_probs.append(probs)

        if store_embs:
            all_embeddings.append(embeddings.detach().cpu())
        all_y.append(y)
        if store_inps:
            all_x.append(x.cpu())
        if store_domains:
            all_domains.append(b[2])

        ### chunking
        if chunking_dir is not None and ((b_idx + 1) % chunk_size == 0 or b_idx == n_batches - 1):
            if store_embs:
                all_embeddings = torch.cat(all_embeddings, axis=0)
            all_y = torch.cat(all_y, axis=0)
            if store_inps:
                all_x = torch.cat(all_x, axis=0)
            if store_domains:
                all_domains = torch.cat(all_domains, axis=0)
            if len(all_probs) > 0:
                all_probs = torch.cat(all_probs, axis=0)
            chunk_files_created.append(os.path.join(chunking_dir, f"chunk_{b_idx + 1}.pt"))
            torch.save({
                "embeddings": all_embeddings,
                "y": all_y,
                "probs": all_probs,
                "x": all_x,
                "domains": all_domains,
            }, chunk_files_created[-1], pickle_module=dill)
            all_x, all_embeddings, all_y, all_probs, all_domains = [], [], [], [], []

    ### final concatenation
    if chunking_dir is not None:
        ### load all chunks and concatenate
        all_embeddings, all_y, all_probs, all_x, all_domains = [], [], [], [], []
        for chunk_file in chunk_files_created:
            chunk = torch.load(chunk_file, pickle_module=dill)
            all_embeddings.append(chunk["embeddings"])
            all_y.append(chunk["y"])
            all_probs.append(chunk["probs"])
            all_x.append(chunk["x"])
            all_domains.append(chunk["domains"])
            os.remove(chunk_file)
        if store_embs:
            all_embeddings = torch.cat(all_embeddings, axis=0)
        all_y = torch.cat(all_y, axis=0)
        all_x = torch.cat(all_x, axis=0) if len(all_x[0]) > 0 else []
        all_domains = torch.cat(all_domains, axis=0) if len(all_domains[0]) > 0 else []
        if len(all_probs[0]) > 0:
            all_probs = torch.cat(all_probs, axis=0)
            if all_probs.ndim == 3:
                acc = [(torch.argmax(all_probs[:,i,:], dim=-1) == all_y).float().mean() for i in range(all_probs.shape[1])]
            else:
                acc = (torch.argmax(all_probs, dim=-1) == all_y).float().mean()
        else:
            all_probs = None
            acc = None
    else:
        if store_embs:
            all_embeddings = torch.cat(all_embeddings, axis=0)
        if store_inps:
            all_x = torch.cat(all_x, axis=0)
        if store_domains:
            all_domains = torch.cat(all_domains, axis=0)
        all_y = torch.cat(all_y, axis=0)
        if len(all_probs) > 0:
            all_probs = torch.cat(all_probs, axis=0)
            if all_probs.ndim == 3:
                acc = [(torch.argmax(all_probs[:,i,:], dim=-1) == all_y).float().mean() for i in range(all_probs.shape[1])]
            else:
                acc = (torch.argmax(all_probs, dim=-1) == all_y).float().mean()
        else:
            all_probs = None
            acc = None
    return all_embeddings, all_y, all_probs, acc, all_x, all_domains


def get_cache_paths(cfg, student_model_name, teacher_model_name, cache_dir_suffix=""):
    cache_path = os.path.join(cfg["save_path"], slugify(cfg['data']['name']) + slugify(cfg['data'].get("type", "")), "cache" + cache_dir_suffix)
    os.makedirs(cache_path, exist_ok=True)

    cached_labels_path = os.path.join(cache_path, f"labels__{teacher_model_name}__{slugify(cfg['data']['name'])}.pt")
    if type(cfg["w2s"]["load_labels"]) == str:
        cached_labels_path = cfg["w2s"]["load_labels"] # can specify full path
    if type(cfg["w2s"]["save_labels"]) == str:
        assert type(cfg["w2s"]["load_labels"]) != str or cfg["w2s"]["load_labels"] == cfg["w2s"]["save_labels"]
        cached_labels_path = cfg["w2s"]["save_labels"] # can specify full path

    cached_embs_path = os.path.join(cache_path, f"embs__{student_model_name.replace('/', '_')}__{slugify(cfg['data']['name'])}.pt")
    if type(cfg["w2s"]["load_embeddings"]) == str:
        cached_embs_path = cfg["w2s"]["load_embeddings"]
    if type(cfg["w2s"]["save_embeddings"]) == str:
        assert type(cfg["w2s"]["load_embeddings"]) != str or cfg["w2s"]["load_embeddings"] == cfg["w2s"]["save_embeddings"]
        cached_embs_path = cfg["w2s"]["save_embeddings"]

    return cached_labels_path, cached_embs_path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # multi-GPU.

        ### deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
