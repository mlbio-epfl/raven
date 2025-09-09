# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import TYPE_CHECKING, List, Optional
from datasets import load_from_disk, DatasetDict
from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer, TextClassificationDataCollator
from transformers import DataCollatorWithPadding
from ...extras import logging
import datasets 

from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy
from .trainer import TextClsTrainer
import sys
import os
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)

def tokenize_dataset(
    raw_ds,
    tokenizer,
    max_ctx: int=1000,
):
    """
    This function prepares the dataset for training. It takes the raw dataset, a formatting function,
    a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """

    def process_function(res):
        toks = tokenizer(res["txt"])
        return dict(
            input_ids=toks["input_ids"],
            hard_label=res['hard_label'] if 'hard_label' in res else res['gt_label'],
            soft_label=res['soft_label'] if 'soft_label' in res else None,
            logits=res['logits'] if "logits" in res else None
        )

    ds = raw_ds.map(process_function, batched=False).filter(lambda x: len(x["input_ids"]) < max_ctx)
    return ds

def run_tc(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    if finetuning_args.loss == 'edl':
        training_args.output_dir += f"_{finetuning_args.gamma}{finetuning_args.lambdas}"
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #template = get_template_and_fix_tokenizer(tokenizer, data_args)
    #dataset_module = get_dataset(template, model_args, data_args, training_args, stage="tc", **tokenizer_module)
    dataset = load_from_disk(data_args.dataset_from_disk)
    if 'val' not in dataset:
        split_dataset = dataset['train'].train_test_split(data_args.val_size, seed=0)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        train_dataset = dataset['train']
        val_dataset = dataset['val']
    dataset_module = {"eval_dataset":val_dataset, "train_dataset":train_dataset}
    test_dataset=dataset['test']
    logger.info_rank0(f"####loaded dataset#### \n {str(test_dataset)}")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, num_labels=data_args.num_labels)
    model.config.pad_token_id = model.config.eos_token_id
    logger.info_rank0(f"####loaded model#### \n {str(model)}")

    #data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    # data_collator = PairwiseDataCollatorWithPadding(
    #     template=template, model=model, pad_to_multiple_of=8, **tokenizer_module
    # )
    data_collator = TextClassificationDataCollator(tokenizer=tokenizer, linear_probe=finetuning_args.linear_probe)
    # Update arguments
    #training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset
    
    training_args.load_best_model_at_end=True
    training_args.greater_is_better = True
    training_args.metric_for_best_model = "eval_accuracy"
    # Initialize our Trainer
    trainer = TextClsTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeAccuracy(),
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])
        
        if finetuning_args.ens:
            save_ens_hist(trainer.ens, training_args.output_dir)
    
    # Evaluation
    if finetuning_args.extract:
        for (eval_dataset, name) in zip([dataset['target'], dataset['val'], dataset['test']],['train', 'val', 'test']):
            predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict")
            new_rows=[]
            for (row, logits) in zip(eval_dataset, predict_results.predictions.tolist()):
                row['logits'] = logits
                new_rows.append(row)
            new_dataset=datasets.Dataset.from_list(new_rows)
            new_dataset.save_to_disk(training_args.output_dir+f'/{name}')

    elif training_args.do_eval:
        # metrics = trainer.evaluate(metric_key_prefix="eval")
        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

        predict_dataset = test_dataset
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict_test")
        trainer.log_metrics("predict_test", predict_results.metrics)
        trainer.save_metrics("predict_test", predict_results.metrics)
        if data_args.save_weak_ds:
            predict_dataset = dataset['target'] if 'target' in dataset else dataset['train']
            predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict_target")
            trainer.log_metrics("predict_target", predict_results.metrics)
            trainer.save_metrics("predict_target", predict_results.metrics)
            new_rows=[]
            for (row, logits) in zip(predict_dataset, predict_results.predictions.tolist()):
                row['logits'] = logits
                new_rows.append(row)
            new_dataset=datasets.Dataset.from_list(new_rows)
            if "target_test" in dataset:
                save_dataset = DatasetDict({
                        "train": new_dataset,
                        "test": dataset['target_test'],
                    })
            else:
                save_dataset = DatasetDict({
                        "train": new_dataset,
                        "val": dataset['val'],
                        "test": dataset['test'],
                    })
            save_dataset.save_to_disk(training_args.output_dir+'/weak_ds')

            
        #trainer.save_predictions(predict_results, test_dataset)
    
    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

def save_ens_hist(ens, save_path):
    save_dict = {}
    weight = ens.history['ensemble_weights']

    value_list=[]
    for w in weight:
        if not (isinstance(w, list)):
            value_list.append(w.tolist()[0])
        else:
            value_list.append(w[0])

    save_dict['ensemble_weights'] = value_list
    loss = ens.history['loss']
    save_dict['loss'] = loss
    os.makedirs(save_path, exist_ok=True)
    with open(save_path+'/ens_weights.json', 'w') as f:
        json.dump(dict(save_dict),f)