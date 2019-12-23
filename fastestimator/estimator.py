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
from fastestimator.util.util import draw, to_list


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
        steps_per_epoch (int, optional): Number of steps to run for each training session. If None, this will be the
            training example number divided by batch_size. (round down). Defaults to None.
        validation_steps (int, optional): Number of steps to run for each evaluation session, If None, this will be
            the evaluation example number divided by batch_size (round down). Defaults to None.
        traces (list, optional): List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps (int, optional): Interval steps of logging. Defaults to 100.
    """
    def __init__(self,
                 pipeline,
                 network,
                 epochs,
                 steps_per_epoch=None,
                 validation_steps=None,
                 traces=None,
                 log_steps=100):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.traces = traces
        self.log_steps = log_steps
        assert log_steps is None or log_steps > 0, "log_steps must be positive or None"
        self.summary = False

    def fit(self, summary=None):
        draw()
        self.summary = summary
        if isinstance(self.pipeline, Pipeline):
            self._prepare_pipeline()
        self._prepare_network()
        self._warmup()
        self._prepare_estimator()
        return self._start()

    def _prepare_pipeline(self):
        pass

    def _prepare_network(self):
        self.network.prepare()

    def _prepare_estimator(self):
        if self.traces is None:
            self.traces = []
        else:
            self.traces = to_list(self.traces)
        