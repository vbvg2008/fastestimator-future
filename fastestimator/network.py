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
import tensorflow as tf
import torch

from fastestimator.op.tensorop.model.update import UpdateOp
from fastestimator.util.util import to_list


class Network:
    """A class representing network operations for FastEstimator model training.
    Args:
        ops : Specifies the series of operations for training model
    """
    def __init__(self, ops):
        self.ops = to_list(ops)

    def run_step(self, batch, ops, state):
        """Execute the ops in Network
        Args:
            batch : dictionary that contains batch data after the pipeline
            ops : operation or list of operations
            state : dictionary that contains meta data
        Returns:
            dictionary containing the predictions of current epoch
        """
        prediction = {}
        batch = ChainMap(prediction, batch)
        mode = state["mode"]
        # use gradient tape for train, otherwise use a dummy tape
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward(batch, state, ops)
        del state['tape']
        del tape
        return prediction

    @staticmethod
    def _forward(batch, state, ops):
        data = None
        for op in ops:
            data = get_inputs_by_op(op, batch, data)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_key(batch, data, op.outputs)


def build(model_def, optimizer_def):
    """build model instance in FastEstimator
    Args:
        model_def (function): function definition that returns tf.keras model or torch.nn.Module
        optimizer_def (function, str): function definition that returns tf.
    Returns:
        models: model(s) compiled by FastEstimator
    """
    models = to_list(model_def())
    if isinstance(optimizer_def, str):
        optimizers = to_list(optimizer_def)
    else:
        optimizers = to_list(optimizer_def())
    assert len(models) == len(optimizers)
    for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
        models[idx] = _fe_compile(model, optimizer)
    if len(models) == 1:
        models = models[0]
    return models


def _fe_compile(model, optimizer):
    #model instance check
    if isinstance(model, tf.keras.Model):
        framework = "tensorflow"
    elif isinstance(model, torch.nn.Module):
        framework = "pytorch"
    else:
        raise ValueError("unrecognized model format: {}".format(type(model)))

    #optimizer auto complete
    if isinstance(optimizer, str):
        optimizer_fn = {
            "tensorflow": {
                'adadelta': tf.optimizers.Adadelta,
                'adagrad': tf.optimizers.Adagrad,
                'adam': tf.optimizers.Adam,
                'adamax': tf.optimizers.Adamax,
                'rmsprop': tf.optimizers.RMSprop,
                'sgd': tf.optimizers.SGD
            },
            "pytorch": {
                'adadelta': torch.optim.Adadelta,
                'adagrad': torch.optim.Adagrad,
                'adam': torch.optim.Adam,
                'adamax': torch.optim.Adamax,
                'rmsprop': torch.optim.RMSprop,
                'sgd': torch.optim.SGD
            }
        }
        if framework == "tensorflow":
            optimizer = optimizer_fn["tensorflow"][optimizer]()
        else:
            optimizer = optimizer_fn["pytorch"][optimizer](params=model.parameters(), lr=0.001)

    #optimizer instance check
    if framework == "tensorflow":
        assert isinstance(optimizer, tf.optimizers.Optimizer)
    else:
        assert isinstance(optimizer, torch.optim.Optimizer)
    model.optimizer = optimizer
    model.fe_compiled = True
    return model
