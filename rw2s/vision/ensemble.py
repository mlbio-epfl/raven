import torch
from torch import nn
from torch.nn import functional as F

from collections import defaultdict



class EnsembleParticipant(nn.Module):
    def __init__(
        self,
        model,
        cfg=None,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)


class Ensemble(nn.Module):
    def __init__(
        self,
        models,
        device,
        ensemble_weights_init="uniform",
        ensemble_weights_per_n_samples=1, # set to num_of_samples to have separate weights per sample
        reset_ensemble_weights_at_epoch_end_to_uniform=False,
        **kwargs,
    ):
        assert len(models) >= 1, "Provide at least one model"
        super().__init__()
 
        self.device = device
        self.models = nn.ModuleList(models)

        self.ensemble_weights_init = ensemble_weights_init
        self.ensemble_weights_per_n_samples = ensemble_weights_per_n_samples
        self.reset_ens_ws_at_epoch_end_to_uniform = reset_ensemble_weights_at_epoch_end_to_uniform
        self.ens_ws = nn.Parameter(torch.empty(self.ensemble_weights_per_n_samples, len(self.models), device=device), requires_grad=False)
        self.reset_ens_ws_kwargs = kwargs
        self.reset_ens_ws(**kwargs)
        self.ens_ws_opter = None
        self.ens_ws_optim_skip_first_epochs = None

        self.history = defaultdict(list)

    def ensemble_weights_update_callback(self, x, y, pred, loss, call_backward=True, keep_history=False, last_in_epoch=False, epoch=None):
        assert self.ens_ws_opter is not None, "Initialize the optimizer of ensemble weights first"

        ### calc grad and update ensemble weights
        if (self.ens_ws_optim_skip_first_epochs is None or
            epoch is None or
            epoch >= self.ens_ws_optim_skip_first_epochs):
            ### update ensemble weights
            if call_backward:
                self.ens_ws_opter.zero_grad()
                loss.backward()
                self.ens_ws_opter.step()
            else:
                self.ens_ws_opter.step()

            ### re-normalize
            with torch.no_grad():
                self.ens_ws.data = self.ens_ws.clamp(min=1e-6)
                self.ens_ws.data = self.ens_ws / self.ens_ws.sum(dim=-1, keepdim=True)
        self.ens_ws_opter.zero_grad()

        ### keep history
        if keep_history:
            self.history["loss"].append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            self.history["ensemble_weights"].append(self.ens_ws.detach().cpu().numpy())

        ### reset ensemble weights at the end of epoch
        if last_in_epoch and self.reset_ens_ws_at_epoch_end_to_uniform:
            self.reset_ens_ws(force_ens_ws_n_samples=self.ens_ws.shape[0], force_init_method="uniform")
            self.ens_ws.requires_grad = True
            self.ens_ws_opter = torch.optim.SGD([self.ens_ws], **self.ensemble_weights_opter_cfg)

        return self.ens_ws

    def add_member(self, model):
        self.models.append(model)

        self.ens_ws = nn.Parameter(torch.empty(self.ensemble_weights_per_n_samples, len(self.models), device=self.device), requires_grad=False)
        self.reset_ens_ws(**self.reset_ens_ws_kwargs)

    def init_ens_ws_optim_run(self, ensemble_weights_opter_cfg, force_ens_ws_n_samples=None, ensemble_weights_optim_skip_first_epochs=None, **kwargs):
        self.reset_ens_ws(force_ens_ws_n_samples=force_ens_ws_n_samples, **kwargs)
        self.history["ensemble_weights"] = [self.ens_ws.detach().cpu().numpy()]
        self.ens_ws.requires_grad = True
        self.ens_ws_opter = torch.optim.SGD([self.ens_ws], **ensemble_weights_opter_cfg)
        self.ens_ws_optim_skip_first_epochs = ensemble_weights_optim_skip_first_epochs
        return self.ens_ws_opter

    @torch.no_grad()
    def reset_ens_ws(self, force_ens_ws_n_samples=None, force_init_method=None, **kwargs):
        ens_ws_n_samples = force_ens_ws_n_samples if force_ens_ws_n_samples is not None else self.ensemble_weights_per_n_samples
        ens_ws_init_method = force_init_method if force_init_method is not None else self.ensemble_weights_init
        if ens_ws_init_method == "uniform":
            self.ens_ws.data = torch.ones(ens_ws_n_samples, len(self.models), device=self.device) / len(self.models)
        elif ens_ws_init_method == "random":
            self.ens_ws.data = torch.rand(ens_ws_n_samples, len(self.models), device=self.device)
            self.ens_ws.data = self.ens_ws / self.ens_ws.sum(dim=-1, keepdim=True)
        elif ens_ws_init_method == "entropy":
            ### set ensemble weights as 1/entropy of predictions
            if "yw" in kwargs:
                ### kwargs["yw"] are soft weak labels (num_samples, num_models, num_classes) -> only compute entropy
                yw = kwargs["yw"].to(self.device)
                entropies = -1 * (yw * yw.log()).sum(dim=-1) # (num_samples, num_models)
            else:
                ### make predictions and compute entropy
                entropies = []
                for m in self.models:
                    if "x" in kwargs:
                        raise NotImplementedError
                        _, y_hat = m(kwargs["x"]) # (batch_size, num_classes)
                        y_hat = F.softmax(y_hat, dim=-1).detach()
                        ent = -1 * (y_hat * y_hat.log()).sum(dim=-1) # (batch_size)
                        entropies.append(ent.mean().item() if ens_ws_n_samples == 1 else ent)
                    elif "yw" in kwargs: # kwargs["yw"] are soft weak labels (num_samples, num_models, num_classes)
                        yw = kwargs["yw"]
                        ent = -1 * (yw * yw.log()).sum(dim=-1) # (num_samples, num_models)
                        
                    elif "dataloader" in kwargs:
                        m_entropies = []
                        for x, y, _ in kwargs["dataloader"]: # calculate entropy per batch and average
                            _, y_hat = m(x)
                            y_hat = F.softmax(y_hat, dim=-1).detach()
                            ent = -1 * (y_hat * y_hat.log()).sum(dim=-1) # (batch_size)
                            m_entropies.append(ent)
                        entropies.append(torch.cat(m_entropies).mean().item() if ens_ws_n_samples == 1 else torch.cat(m_entropies))
                    else:
                        raise ValueError("Provide data to compute entropy of predictions")
                if ens_ws_n_samples == 1:
                    entropies = torch.tensor(entropies, device=self.device)[None,:].expand(ens_ws_n_samples, -1) # (num_models) -> (1, num_models)
                else:
                    entropies = torch.stack(entropies).T # (num_samples, num_models)

            ### set weights as 1/entropy
            self.ens_ws.data = 1 / (entropies + 1e-8)
            self.ens_ws.data = self.ens_ws / self.ens_ws.sum(dim=-1, keepdim=True) # normalize
        else:
            raise ValueError(f"Invalid ensemble_weights_init: {ens_ws_init_method}")

    def combine_logits(self, y_hats, x=None, pred=None, force_device=False, sample_idxs=None, treat_as_probas=False, epoch=None):
        """ y_hats: list of tensors of shape (batch_size, num_classes) or (batch_size, num_models, num_classes) """
        if not isinstance(y_hats, torch.Tensor):
            y_hats = torch.stack(y_hats, dim=1)

        if y_hats.ndim == 2:
            y_hats = y_hats[:,None,:] # (batch_size, 1, num_classes)

        ### sample_idxs: (start, end) -> select samples from ensemble weights
        if sample_idxs is not None and self.ens_ws.shape[0] > 1:
            ens_ws = self.ens_ws[sample_idxs,:,None]
        else:
            ens_ws = self.ens_ws[:,:,None]

        if force_device:
            new_y_hats = (y_hats * ens_ws.to(y_hats.device)).sum(dim=1)
        else:
            new_y_hats = (y_hats * ens_ws).sum(dim=1) # (batch_size, num_classes)

        if treat_as_probas:
            new_y_hats = new_y_hats / new_y_hats.sum(dim=-1, keepdim=True) # sum to 1

        return (new_y_hats, pred)

    def forward(self, x, combine_logits=True, collect_embeddings=True):
        embs, y_hats = [], []
        for m in self.models:
            emb, y_hat = m(x)
            if collect_embeddings:
                embs.append(emb)
            y_hats.append(y_hat)
        if collect_embeddings:
            embs = torch.stack(embs, dim=1) # (batch_size, num_models, emb_dim)
        return embs, self.combine_logits(y_hats) if combine_logits else torch.stack(y_hats, dim=1)


class EnsembleDisagreementSchedule:
    def __init__(self, apply_first_n_epochs):
        """ Initialize the disagreement schedule
        Parameters:
            apply_first_n_epochs : int : number of epochs to apply disagreement schedule
        """
        self.apply_first_n_epochs = apply_first_n_epochs

    def get_sample_weights(self, sample_ws, yw):
        """ Get sample weights for the current epoch by masking sample weights based on disagreement between models
        Parameters:
            sample_ws : torch.Tensor(num_samples,) : sample weights to mask
            yw : torch.Tensor(num_samples, num_models, num_classes) : predictions of num_models models
        Returns:
            sample_ws : torch.Tensor(num_samples,) : masked sample weights
        """
        yw_argmax = yw.argmax(dim=-1) # (num_samples, num_models)
        all_agree_mask = (yw_argmax == yw_argmax[:,0,None]).all(dim=-1) # (num_samples)
        return sample_ws * all_agree_mask

    def __call__(self, sample_ws, yw, epoch):
        """ Get sample weights for the current epoch
        Parameters:
            sample_ws : torch.Tensor(num_samples,) : sample weights to mask
            yw : torch.Tensor(num_samples, num_models, num_classes) : predictions of num_models models
            epoch : int : current epoch
        Returns:
            sample_ws : torch.Tensor(num_samples,) : masked sample weights
        """
        if epoch < self.apply_first_n_epochs:
            return self.get_sample_weights(sample_ws, yw)
        return sample_ws
