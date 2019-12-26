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
import time

import numpy as np


class Trace:
    """Trace controls the training loop. User can use `Trace` to customize their own operations.
    Args:
        inputs (str, list, set): A set of keys that this trace intends to read from the state dictionary as inputs
        outputs (str, list, set): A set of keys that this trace intends to write into the state dictionary
        mode (string): Restrict the trace to run only on given modes ('train', 'eval'). None will always
                        execute
    """
    def __init__(self, inputs=None, mode=None, log_names=None):
        self.mode = mode
        self.inputs = inputs
        self.log_names = log_names
        self.system = None

    def on_begin(self):
        """Runs once at the beginning of training
        """
    def on_epoch_begin(self):
        """Runs at the beginning of each epoch
        """
    def on_batch_begin(self):
        """Runs at the beginning of each batch
        """
    def on_batch_end(self, data):
        """Runs at the end of each batch

        Args:
            data: value fetched by the inputs
        """
    def on_epoch_end(self):
        """Runs at the end of each epoch
        """
    def on_end(self, state):
        """Runs once at the end training.
        """


class TrainEssential(Trace):
    """Essential training information for logging during training. Please don't add this trace into an estimator
    manually. An estimator will add it automatically.
    """
    def __init__(self):
        super().__init__(mode="train", log_names=["examples/sec", "progress", "total_time"])
        self.elapse_times = []
        self.num_example = 0
        self.time_start = None
        self.train_start = None
        self.system = None

    def on_begin(self):
        self.train_start = time.perf_counter()

    def on_epoch_begin(self):
        self.time_start = time.perf_counter()

    def on_batch_end(self, data):
        self.num_example += self.system.batch_size
        if self.system.global_step % self.system.log_steps == self.system.log_steps - 1:
            self.elapse_times.append(time.perf_counter() - self.time_start)
            self.system.add_buffer("examples/sec", round(self.num_example / np.sum(self.elapse_times), 1))
            self.system.add_buffer("progress", "{:.1%}".format(self.system.global_step / self.system.total_steps))
            self.elapse_times = []
            self.num_example = 0
            self.time_start = time.perf_counter()

    def on_epoch_end(self):
        self.elapse_times.append(time.perf_counter() - self.time_start)

    def on_end(self, state):
        self.system.add_buffer("total_time", "{} sec".format(round(time.perf_counter() - self.train_start, 2)))


class Logger(Trace):
    def __init__(self, log_names):
        super().__init__()
        self.system = None

    