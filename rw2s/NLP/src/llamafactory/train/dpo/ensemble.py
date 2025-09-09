import torch
from torch import nn
from torch.nn import functional as F

from collections import defaultdict


class Ensemble(nn.Module):
    def __init__(
        self,
        n_models,
        device,
        ensemble_weights_init="uniform",
        ensemble_weights_per_n_samples=1, # set to num_of_samples to have separate weights per sample
        reset_ensemble_weights_at_epoch_end_to_uniform=True,
        easy_hard_schedule=0.1,
        **kwargs,
    ):
        assert n_models >= 1, "Provide at least one model"
        super().__init__()
        self.easy_hard_schedule = easy_hard_schedule
        self.n_models= n_models
        self.device = device
        self.ensemble_weights_init = ensemble_weights_init
        self.ensemble_weights_per_n_samples = ensemble_weights_per_n_samples
        self.reset_ens_ws_at_epoch_end_to_uniform = reset_ensemble_weights_at_epoch_end_to_uniform
        self.ens_ws = nn.Parameter(torch.empty(self.ensemble_weights_per_n_samples, n_models, device=device), requires_grad=False)
        self.reset_ens_ws_kwargs = kwargs
        self.reset_ens_ws(**kwargs)
        self.ens_ws_opter = None
        self.ens_ws_optim_skip_first_epochs = None
        self.eval=False
        self.history = defaultdict(list)

    def ensemble_weights_update_callback(self, loss, call_backward=True, keep_history=False, last_in_epoch=False, epoch=None):
        assert self.ens_ws_opter is not None, "Initialize the optimizer of ensemble weights first"
        if not loss.requires_grad:
            #print('eval')
            return

        ### calc grad and update ensemble weights
        if (self.ens_ws_optim_skip_first_epochs is None or
            epoch is None or
            epoch >= self.ens_ws_optim_skip_first_epochs):
            ### update ensemble weights
            if call_backward:
                self.ens_ws_opter.zero_grad()
                loss.backward(retain_graph=True)
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
            self.ens_ws.data = torch.ones(ens_ws_n_samples, self.n_models, device=self.device) / self.n_models
        elif ens_ws_init_method == "random":
            self.ens_ws.data = torch.rand(ens_ws_n_samples, self.n_models, device=self.device)
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
        elif ens_ws_init_method == "kl_disagreement":
            assert force_ens_ws_n_samples is None, "Cannot set ensemble_weights_per_n_samples with kl_disagreement initialization (not implemented)"

            ### set ensemble weights as the average of KL divergence between each pair of models
            kls, n_samples = torch.zeros(len(self.models), device=self.device), 0
            for x, y, _ in kwargs["dataloader"]:
                for i, m1 in enumerate(self.models):
                    y_hat1 = F.log_softmax(m1(x)[1].detach(), dim=-1)
                    kl_batch = 0.
                    for j, m2 in enumerate(self.models):
                        y_hat2 = F.log_softmax(m2(x)[1].detach(), dim=-1)
                        kl_batch += F.kl_div(y_hat1, y_hat2, reduction="sum", log_target=True)
                    kls[i] += kl_batch / (len(self.models) - 1) # average over all other models
                n_samples += x.size(0)
            kls /= n_samples
            self.ens_ws.data = kls[None,:].expand(ens_ws_n_samples, -1).sum() / kls[None,:].expand(ens_ws_n_samples, -1)  # <=> 1 / (kls / kls.sum())
            self.ens_ws.data = self.ens_ws / self.ens_ws.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Invalid ensemble_weights_init: {ens_ws_init_method}")

    def combine_logits(self, y_hats, force_device=False, sample_idxs=None, treat_as_probas=False, epoch=None):
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
            return (y_hats * ens_ws.to(y_hats.device)).sum(dim=1)

        new_y_hats = (y_hats * ens_ws).sum(dim=1) # (batch_size, num_classes)
        if treat_as_probas:
            return new_y_hats / new_y_hats.sum(dim=-1, keepdim=True) # sum to 1
        return new_y_hats

    def forward(self, tensor_data):

        out = (tensor_data*self.ens_ws.unsqueeze(0)).sum(-1)
        out = torch.nn.functional.softmax(out, dim=-1)
        return out

