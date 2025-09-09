########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import json
import datasets



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=5e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    target_set_path: Optional[str] = field(
        default="RLHFlow/HH-RLHF-Helpful-standard",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="./bt_models/qwen",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "seed"},
    )
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
# tokenizer.padding_side = "right"
###

# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path
target_train_path = script_args.target_set_path

def build_dataset(tokenizer, train_path, eval_path):

    def tokenize(sample):
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False)
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False)
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    ds = load_dataset(train_path, split="train").shuffle(seed=script_args.seed) #42
    #ds = ds.select(range(2000))
    ds = ds.map(tokenize, num_proc=8)

    eval_dataset = None

    split_data = ds.train_test_split(0.1, seed=0)
    train_dataset, eval_dataset = split_data['train'], split_data['test']
    print('train', train_dataset)
    print('eval', eval_dataset)
    return train_dataset, eval_dataset


train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

# Define the trainer


# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb',
    load_best_model_at_end=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True,
)


model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)


trainer.train()
metrics = trainer.evaluate()


print("Saving last checkpoint of the model")
#model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
with open(output_name + f"/ID_{metrics['eval_accuracy']}.json", 'w') as f:
    json.dump(metrics, f, indent=2,ensure_ascii=False)


target_train_dataset, _ = build_dataset(tokenizer, target_train_path, eval_path)
pred = trainer.predict(target_train_dataset)
with open(output_name + f"/OOD_{pred.metrics['test_accuracy']}.json", 'w') as f:
    json.dump(pred.metrics, f, indent=2,ensure_ascii=False)

finetuning_data = []
for p in zip(target_train_dataset, pred.predictions[0], pred.predictions[1]):
    data, pred1, pred2 = p
    if pred2 > pred1:
        new_data = {}
        new_data['chosen'] = data['rejected']
        new_data['rejected'] = data['chosen']
        new_data['positive'] = data['negative']        
        new_data['negative'] = data['positive']
        new_data['input_ids_j'] = data['input_ids_k']
        new_data['input_ids_k'] = data['input_ids_j']
        new_data['attention_mask_j'] = data['attention_mask_k']
        new_data['attention_mask_k'] = data['attention_mask_j']
        finetuning_data.append(new_data)
    else:
        finetuning_data.append(data)
    
new_dataset = datasets.Dataset.from_list(finetuning_data)
new_dataset.save_to_disk(output_name + '/target_train')