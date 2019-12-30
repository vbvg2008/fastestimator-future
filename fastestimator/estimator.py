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
from fastestimator.op.op import get_inputs_by_key
from fastestimator.op.tensorop.model import UpdateOp
import pdb

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
        self.do_eval = False

    def fit(self):
        draw()
        self._prepare_estimator()
        self._prepare_network()
        self._prepare_pipeline()
        return self._start()

    def _prepare_pipeline(self):
        if isinstance(self.pipeline.train_data, tf.data.Dataset):
            assert self.steps_per_epoch, "must provide steps_per_epoch expicity when using tensorflow Dataset for training"
        elif self.steps_per_epoch is None:
            self.steps_per_epoch = len(self.pipeline)
        self.system.total_steps = self.epochs * self.steps_per_epoch
        self.do_eval = bool(self.pipeline.eval_data)

    def _prepare_network(self):
        self.network.exported_keys = self.network.op_outputs.intersection(self.trace_inputs)

    def _prepare_estimator(self):
        self._prepare_traces()
        self._prepare_system()

    def _prepare_system(self):
        self.system = System(mode="train",
                             global_step=0,
                             batch_size=None,
                             num_devices=get_num_devices(),
                             log_steps=self.log_steps,
                             total_epochs=self.epochs,
                             total_steps=None,
                             epoch_idx=0,
                             batch_idx=0)
        for trace in self.traces:
            trace.system = self.system

    def _prepare_traces(self):
        if self.traces is None:
            self.traces = []
        self.traces = to_list(self.traces)
        self.traces.insert(0, TrainEssential())
        self.monitor_names = set(filter(None, to_list(self.monitor_names)))
        self._initialize_trace_inputs()
        for trace in self.traces:
            self.trace_inputs = self.trace_inputs.union(set(filter(None, to_list(trace.inputs))))
            self.monitor_names = self.monitor_names.union(set(filter(None, to_list(trace.log_names))))
        self.traces.append(Logger(self.monitor_names))

    def _initialize_trace_inputs(self):
        for op in self.network.ops:
            if isinstance(op, UpdateOp):
                self.trace_inputs = self.trace_inputs.union(set(to_list(op.inputs)))

    def _start(self):
        self._run_traces_on_begin()
        for self.system.epoch_idx in range(self.epochs):
            self.system.mode = "train"
            self._run_epoch()
            if self.do_eval:
                self.system.mode = "eval"
                self._run_epoch()
        self._run_traces_on_end()
    
    def _run_epoch(self):
        self._run_traces_on_epoch_begin()
        ds_iter = self.pipeline.get_iterator(self.system.mode)
        for self.system.batch_idx, batch in enumerate(ds_iter):
            self.system.batch_size = self.pipeline.get_batch_size(self.system.epoch_idx)
            self._run_traces_on_batch_begin()
            prediction = self.network.run_step(batch, {"mode": self.system.mode, "epoch": self.system.epoch_idx})
            self._run_traces_on_batch_end(batch, prediction)
            self.system.update_global_step()
        self._run_traces_on_epoch_end()
        self.system.update_epoch_idx()

    def _run_traces_on_begin(self):
        for trace in self.traces:
            trace.on_begin()
        self.system.clear_buffer()

    def _run_traces_on_epoch_begin(self):
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                trace.on_epoch_begin()
        self.system.clear_buffer()

    def _run_traces_on_batch_begin(self):
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                trace.on_batch_begin()
        self.system.clear_buffer()

    def _run_traces_on_batch_end(self, batch, prediction):
        batch = ChainMap(prediction, batch)
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                if trace.inputs:
                    data = get_inputs_by_key(batch, trace.inputs)
                else:
                    data = None
                trace.on_batch_end(data)
        self.system.clear_buffer()
    
    def _run_traces_on_epoch_end(self):
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                trace.on_epoch_end()
        self.system.clear_buffer()
    
    def _run_traces_on_end(self):
        for trace in self.traces:
            trace.on_end()
        self.system.clear_buffer()


class System:
    def __init__(self, mode, global_step, batch_size, num_devices, log_steps, total_epochs, total_steps, epoch_idx, batch_idx):
        self.mode = mode
        self.global_step = global_step
        self.batch_size = batch_size
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

    def update_epoch_idx(self):
        if self.mode == "train":
            self.epoch_idx += 1
    
    def update_global_step(self):
        if self.mode == "train":
            self.global_step += 1