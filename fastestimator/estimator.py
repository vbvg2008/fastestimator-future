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
from collections import ChainMap

import tensorflow as tf
import torch

from fastestimator.pipeline import Pipeline
from fastestimator.trace.trace import Logger, TrainEssential
from fastestimator.util.util import draw, get_num_devices, to_list


class Estimator:
    """Estimator is the highest level class that user can directly use for traning a model (estimator.fit). It wraps
    up `Pipeline`, `Network`, `Trace` objects together and defines the whole optimization process with other training
    necessary information.
    Args:
        pipeline (obj): Pipeline object that defines the data processing workflow. It should be an instance of
            `fastestimator.pipepline.pipeline.Pipeline`
        network (obj): Network object that defines models and their external connection. It should be an instance of
            `fastestimator.network.network.Network`
        epochs (int): Number of epooch to run.
        steps_per_epoch (int, optional): Number of steps to run for each epoch.
        traces (list, optional): List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps (int, optional): Interval steps of logging. Defaults to 100.
        monitor_names (str, list): Additional keys to print in logger
    """
    def __init__(self, pipeline, network, epochs, steps_per_epoch=None, traces=None, log_steps=100, monitor_names=None):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.traces = traces
        self.log_steps = log_steps
        assert log_steps is None or log_steps > 0, "log_steps must be positive or None"
        self.monitor_names = monitor_names
        self.trace_inputs = set()

    def fit(self):
        draw()
        self._prepare_estimator()
        self._prepare_network()
        self._prepare_pipeline()
        return self._start()

    def _prepare_pipeline(self):
        assert isinstance(self.pipeline, (tf.data.Dataset, Pipeline, torch.utils.data.DataLoader)), "please provide \
            one of the following: fe.Pipeline, tf.data.Dataset or torch.utils.data.DataLoader"

        if isinstance(self.pipeline, tf.data.Dataset):
            assert self.steps_per_epoch, "must provide steps_per_epoch expicity when using tensorflow Dataset"
        elif self.steps_per_epoch is None:
            self.steps_per_epoch = len(self.pipeline)
        self.system.total_steps = self.epochs * self.steps_per_epoch

    def _prepare_network(self):
        self.network.exported_keys = self.network.op_outputs.intersection(self.trace_inputs)

    def _prepare_estimator(self):
        self._prepare_system()
        self._prepare_traces()

    def _prepare_system(self):
        self.system = System(mode="train",
                             global_step=0,
                             num_devices=get_num_devices(),
                             log_steps=self.log_steps,
                             total_epochs=self.epochs,
                             total_steps=None,
                             epoch_idx=0,
                             batch_idx=0)

    def _prepare_traces(self):
        if self.traces is None:
            self.traces = []
        self.traces = to_list(self.traces)
        self.traces.insert(0, TrainEssential())
        self.monitor_names = set(filter(None, to_list(self.monitor_names)))
        for trace in self.traces:
            self.trace_inputs = self.trace_inputs.union(set(filter(None, to_list(trace.inputs))))
            self.monitor_names = self.monitor_names.union(set(filter(None, to_list(trace.log_names))))
        self.traces.append(Logger(self.monitor_names))

    def _start(self):
        pass


class System:
    def __init__(self, mode, global_step, num_devices, log_steps, total_epochs, total_steps, epoch_idx, batch_idx):
        self.mode = mode
        self.global_step = global_step
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.buffer = {}

    def add_buffer(self, key, value):
        self.buffer[key] = value

    def clear_buffer(self):
        del self.buffer
        self.buffer = {}
