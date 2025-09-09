# Copyright 2024 the LlamaFactory team.
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r"""
    Computes reward accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        if isinstance(preds, tuple):
            ens_acc = np.mean(labels==preds[0].logits.argmax(1)) 
            def_acc = np.mean(labels==preds[1].logits.argmax(1)) 
            return {"accuracy": ens_acc, "def_accuracy": def_acc}
        else:
            accuracy= np.mean(labels==preds.argmax(1))
            return {"accuracy": accuracy}
