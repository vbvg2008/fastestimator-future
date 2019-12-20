# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
# ==============================================================================
from fastestimator.backend.cross_entropy import cross_entropy
from fastestimator.op.op import TensorOp


class CrossEntropy(TensorOp):
    """Calculate Element-Wise CrossEntropy(binary, categorical or sparse categorical)

    Args:
        inputs: A tuple or list like: [<y_pred>, <y_true>]
        outputs: key to store the computed loss value (not required under normal use cases)
        mode: 'train', 'eval', 'test', or None
    """
    def __init__(self, inputs=None, outputs=None, mode=None, apply_softmax=False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.apply_softmax = apply_softmax

    def forward(self, data, state):
        y_pred, y_true = data
        loss = cross_entropy(y_pred, y_true, apply_softmax=self.apply_softmax)
        return loss
