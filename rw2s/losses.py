"""
From github.com/openai/weak-to-strong.git
"""
import torch
import torch.nn.functional as F


class LossFnBase:
    def apply_reduction(self, loss, reduction):
        """
        This function applies the reduction to the loss.

        Parameters:
        loss: The loss tensor.
        reduction: The reduction type.

        Returns:
        The reduced loss tensor.
        """
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def __call__(
        self,
        logits,
        labels,
        **kwargs,
    ):
        """
        This function calculates the loss between logits and labels.
        """
        raise NotImplementedError


class xent_loss(LossFnBase):
    def __call__(
        self,
        logits,
        labels,
        step_frac=None,
        reduction="mean",
        sample_weights=None,
    ):
        """
        This function calculates the cross entropy loss between logits and labels.

        Parameters:
        logits: The predicted values.
        labels: The actual values.
        step_frac: The fraction of total training steps completed.
        reduction: The reduction type.
        sample_weights: The weights for each sample.

        Returns:
        The mean of the cross entropy loss.
        """
        if labels.ndim == 3 and labels.shape[-1] == logits.shape[-1]:
            loss = 0.
            for m_i in range(labels.shape[1]):
                loss += torch.nn.functional.cross_entropy(logits, labels[:,m_i], reduction="none")
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        if sample_weights is not None:
            assert sample_weights.shape[0] == loss.shape[0]
            loss = loss * sample_weights

        return self.apply_reduction(loss, reduction)


class logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence.

    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef=0.5,
        warmup_frac=0.2,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits,
        labels,
        step_frac,
        sample_weights=None,
        reduction="mean",
    ):
        logits = logits.float()
        labels = labels.float()
        # coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = 1.0 if step_frac >= self.warmup_frac else (step_frac / self.warmup_frac)
        coef = coef * self.aux_coef

        strong_preds = torch.argmax(logits, dim=-1).detach()
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=labels.shape[-1])
        if labels.ndim == 3 and labels.shape[-1] == strong_preds.shape[-1]:
            target = labels * (1 - coef) + strong_preds.unsqueeze(1) * coef
            loss = 0.
            for m_i in range(target.shape[1]):
                loss += torch.nn.functional.cross_entropy(logits, target[:,m_i], reduction="none")
        else:
            target = labels * (1 - coef) + strong_preds * coef
            loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")

        if sample_weights is not None:
            assert sample_weights.shape[0] == loss.shape[0]
            loss = loss * sample_weights

        return self.apply_reduction(loss, reduction)


class adapt_logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence with adaptive alpha parameter.
    """

    def __call__(
        self,
        logits,
        labels,
        step_frac,
        sample_weights=None,
        reduction="mean",
    ):
        logits = logits.float()
        labels = labels.float()

        ### compute adaptive alpha coef
        strong_preds = torch.argmax(logits, dim=1).detach()
        ce_self = torch.exp(torch.nn.functional.cross_entropy(logits, strong_preds, reduction="none"))
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=labels.shape[-1])

        if labels.ndim == 3 and labels.shape[-1] == strong_preds.shape[-1]:
            loss = 0.
            for m_i in range(labels.shape[1]):
                ce_teacher = torch.exp(torch.nn.functional.cross_entropy(logits, torch.argmax(labels[:,m_i], dim=-1), reduction="none"))
                alpha = (ce_self / (ce_self + ce_teacher)).detach()[:,None]
                target = labels[:,m_i] * (1 - alpha) + strong_preds * alpha
                loss += torch.nn.functional.cross_entropy(logits, target, reduction="none")
        else:
            ce_teacher = torch.exp(torch.nn.functional.cross_entropy(logits, torch.argmax(labels, dim=-1), reduction="none"))
            alpha = (ce_self / (ce_self + ce_teacher)).detach()[:,None]
            target = labels * (1 - alpha) + strong_preds * alpha
            loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")

        if sample_weights is not None:
            assert sample_weights.shape[0] == loss.shape[0]
            loss = loss * sample_weights

        return self.apply_reduction(loss, reduction)


class edl_log_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for the Evidential Deep Learning-based loss
    proposed by Cui Z. et al. 2024 (https://arxiv.org/abs/2406.03199) 

    Attributes:
    gamma: A float indicating the auxiliary coefficient for balancing the loss coming from the student self-supervision and teachers.
    lambdas: Weights for the losses coming from different weak models (teachers). \lambda_i in Eq. 4.
    """

    def __init__(
        self,
        gamma=0.5,
        lambdas=1,
    ):
        self.gamma = gamma
        self.lambdas = lambdas

    @staticmethod
    def kl_divergence(alpha, num_classes):
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
            + torch.lgamma(ones).sum(dim=-1, keepdim=True)
            - torch.lgamma(ones.sum(dim=-1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=-1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    @staticmethod
    def edl_log_loss(output, target, step_frac):
        alpha = F.relu(output) + 1 # evidence + 1 # (B, num_classes)

        ### NLL
        S = torch.sum(alpha, dim=-1, keepdim=True) # (B, 1)
        A = torch.sum(target * (torch.log(S) - torch.log(alpha)), dim=-1, keepdim=True) # (B, 1)

        ### regularization
        kl_alpha = (alpha - 1) * (1 - target) + 1 # y + (1 - y) * alpha
        kl_div = step_frac * edl_log_loss_fn.kl_divergence(kl_alpha, num_classes=target.shape[-1])

        return A + kl_div

    def __call__(
        self,
        logits,
        labels,
        step_frac,
        sample_weights=None,
        reduction="mean",
    ):
        if labels.ndim == 2:
            labels = labels.unsqueeze(1)
        if not type(self.lambdas) in (int, float):
            assert len(self.lambdas) == labels.shape[1], "Number of lambdas should match number of teachers"

        num_classes = labels.shape[-1]
        logits = logits.float()
        labels = labels.float() # soft labels (B, num_teachers, num_classes)

        ### compute EDL loss wrt to argmax'ed student labels
        student_onehot = F.one_hot(torch.argmax(logits, dim=-1), num_classes=num_classes) # (B, num_classes)
        edl_student = edl_log_loss_fn.edl_log_loss(logits, student_onehot, step_frac=step_frac).squeeze(-1) # (B,)

        ### compute EDL loss wrt to soft teacher labels
        teacher_onehot = F.one_hot(torch.argmax(labels, dim=-1), num_classes=num_classes) # (B, num_teachers, num_classes)
        edl_teachers = 0
        for m in range(labels.shape[1]):
            edl_curr_teacher = edl_log_loss_fn.edl_log_loss(logits, teacher_onehot[:, m], step_frac=step_frac) # (B, 1)
            # eq.4: multiply by soft labels
            edl_curr_teacher = (labels[:,m] * edl_curr_teacher.expand(-1, num_classes)).sum(dim=-1) # (B,)
            edl_teachers += self.lambdas * edl_curr_teacher if type(self.lambdas) in (int, float) else self.lambdas[m] * edl_curr_teacher

        ### combine (eq.5)
        loss = edl_teachers * (1 - self.gamma) + edl_student * self.gamma
        if sample_weights is not None:
            assert sample_weights.shape[0] == loss.shape[0]
            loss = loss * sample_weights

        return self.apply_reduction(loss, reduction)


LOSS_DICT = {
    "logconf": logconf_loss_fn,
    "adapt_logconf": adapt_logconf_loss_fn,
    "xent": xent_loss,
    "edl": edl_log_loss_fn,
}
