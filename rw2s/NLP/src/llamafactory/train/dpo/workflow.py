# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer, EnsembleDataCollatorWithPadding
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer,EnsembleDPOTrainer
import json
import os
import sys
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments
import sys
from datasets import load_from_disk
import datasets
def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    if data_args.dataset_from_disk:
        dataset_module = {"eval_dataset":load_from_disk(os.path.join(data_args.dataset_from_disk,'eval_dataset')), 
                          "train_dataset":load_from_disk(os.path.join(data_args.dataset_from_disk,'train_dataset'))}
    else:
        dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", easy_hard=finetuning_args.easy_hard, **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    print('ens###########', finetuning_args.ens)
    if finetuning_args.ens:
        collator = EnsembleDataCollatorWithPadding
        trainer_class = EnsembleDPOTrainer
        #easy_dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    else:
        collator = PairwiseDataCollatorWithPadding
        trainer_class = CustomDPOTrainer

    data_collator = collator(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    # # Initialize our Trainer
    # trainer = CustomDPOTrainer(
    #     model=model,
    #     ref_model=ref_model,
    #     args=training_args,
    #     finetuning_args=finetuning_args,
    #     data_collator=data_collator,
    #     callbacks=callbacks,
    #     **dataset_module,
    #     **tokenizer_module,
    # )

    if finetuning_args.edl:
        for dataset_split in ["eval_dataset", "train_dataset"]:
            predict_dataset = dataset_module[dataset_split] #dataset_module["train_dataset"]
            print(predict_dataset)
            pred = trainer.predict(predict_dataset)
            print(f'############{pred.predictions.shape}########## ')
            print(f'############{pred.predictions[0],pred.predictions[pred.predictions.shape[0]//2]}########## ')
            save_data = []
            for i, z in enumerate(zip(pred.predictions, predict_dataset)):
                logprobs, data = z
                data['logprobs'] = [logprobs, pred.predictions[i+pred.predictions.shape[0]//2]]
                save_data.append(data)
            new_dataset = datasets.Dataset.from_list(save_data)
            new_dataset.save_to_disk(training_args.output_dir+f'/logprobs_{dataset_split}')
    else:
        # Training
        if training_args.do_train:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            if finetuning_args.include_effective_tokens_per_second:
                train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                    dataset_module["train_dataset"], train_result.metrics, stage="rm"
                )

            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

        if finetuning_args.ens:
            save_ens_hist(trainer.ens, training_args.output_dir)

        # Evaluation
        if training_args.do_eval:
            # if finetuning_args.ens:
            #     trainer.ens.eval=True
            metrics = trainer.evaluate(metric_key_prefix="eval")
            if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
                remove_keys = [key for key in metrics.keys() if "rewards" in key]
                for key in remove_keys:
                    metrics.pop(key)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

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