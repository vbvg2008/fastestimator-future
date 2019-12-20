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
from fastestimator.backend.update import update_model
from fastestimator.op import TensorOp


class UpdateOp(TensorOp):
    """This class performs updates to a model's weights based on the loss

    Args:
        model (tf.keras.Model or torch.nn.Module): model instance compiled by fe.build
        loss (str): the name of loss
    """
    def __init__(self, model, loss_name, mode="train"):
        super().__init__(inputs=loss_name, outputs=None, mode=mode)
        self.model = model

    def forward(self, data, state):
        loss = data
        update_model(self.model, loss, tape=state['tape'])