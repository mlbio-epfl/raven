import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import torch.nn as nn

class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, pretrained_model_name_or_path, linear_probe=True, std = 0, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels

        lm = None
        if linear_probe:
            self.transformer = nn.Identity()
        else:
            lm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
            try:
                self.transformer = lm.transformer
            except:
                self.transformer = lm.model

        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            lm.lm_head.weight.dtype if lm else torch.float32
        )
        torch.nn.init.normal_(self.score.weight, std=std)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls(pretrained_model_name_or_path, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def get_features(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        return hidden_states

    def forward(self, inp, output_features = False, linear_probe=True):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        if linear_probe:
            inp = inp.to(self.score.weight.device)
            return SequenceClassifierOutputWithPast(loss=None, logits=self.score(inp))
        hidden_states = self.get_features(inp)
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        if output_features:
            return logits, hidden_states
        
        return logits
    
    
class MixtureOfWeakModels(PreTrainedModel):
    """
    "Mixture of weak models for the ICML rebuttal response. Weak models are 
    """

    def __init__(self, pretrained_model_name_or_path, num_weak_models=3, num_labels=5, weak_logits=True, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        super().__init__(config)
        
        if 'llama' in pretrained_model_name_or_path.lower():
            strong_feature_size = 4096
            weak_feature_size = 2048
        elif 'qwen' in pretrained_model_name_or_path.lower():
            strong_feature_size = 3584
            weak_feature_size = 896
        else:
            raise Exception("The model should be either qwen or llama")

        self.num_labels = num_labels
        self.weak_logits=weak_logits
        self.selector = torch.nn.Linear(strong_feature_size, num_weak_models, bias=False)
        if not weak_logits:
            self.classifiers = nn.ModuleList(
                [torch.nn.Linear(weak_feature_size, self.num_labels) for _ in range(num_weak_models)]
            )
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls(pretrained_model_name_or_path, **kwargs)

    def forward(self, strong_feature, weak, ensemble=True):

        strong_feature = strong_feature.to(self.selector.weight.device)
        ensemble_weights = self.selector(strong_feature)

        if self.weak_logits:
            outputs = weak
        else:

            outputs = []
            for i, classifier in enumerate(self.classifiers):
                # Select the features corresponding to the i-th weak model: shape (batch_size, in_features)
                feature = weak
                # Pass through the i-th classifier to get (batch_size, out_features)
                outputs.append(classifier(feature))
        
            # Optionally, stack outputs to form a tensor of shape (batch_size, num_models, out_features)
            outputs = torch.stack(outputs, dim=1)
        logits = (outputs*ensemble_weights.unsqueeze(-1)).sum(1)
        if ensemble:
            return SequenceClassifierOutputWithPast(loss=None, logits=logits)
        
        else:
            ensemble_weights = torch.nn.functional.one_hot(ensemble_weights.argmax(1), num_classes=3)
        #print(outputs.shape, ensemble_weights.shape)
        defer_logits = (outputs*ensemble_weights.unsqueeze(-1)).sum(1)

        return {'ensemble': SequenceClassifierOutputWithPast(loss=None, logits=logits),
                'defer': SequenceClassifierOutputWithPast(loss=None, logits=defer_logits)}