# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for training an MPS classifier using automatic differentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import time
from typing import Tuple, Optional
from experiments.MPS_classifier import classifier


def run_step(mps: classifier.MatrixProductState, optimizer,
             data: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor]:
  """Runs a single training step for one batch"""
  with tf.GradientTape() as tape:
    tape.watch(mps.tensors)
    loss, logits = mps.loss(data, labels)

  grads = tape.gradient(loss, mps.tensors)
  optimizer.apply_gradients(zip(grads, mps.tensors))

  return loss, logits


def run_epoch(mps: classifier.MatrixProductState,
              data_generator,
              n_batch: int, optimizer=None) -> Tuple[tf.Tensor]:
  """Performs a whole training epoch.
  One epoch corresponds to a full iteration over the training set.
  """
  loss, logits = 0.0, []
  for _ in range(n_batch):
    data, labels = next(data_generator)
    if optimizer is None:
      batch_results = mps.loss(data, labels)
    else:
      batch_results = run_step(mps, optimizer, data, labels)
    loss += batch_results[0]
    logits.append(batch_results[1])

  return loss, tf.concat(logits, axis=0)


def fit(mps: classifier.MatrixProductState, optimizer,
        x: tf.Tensor, y: tf.Tensor,
        x_val: Optional[tf.Tensor] = None,
        y_val: Optional[tf.Tensor] = None,
        n_epochs: int = 20, batch_size: int = 10,
        n_message: int = 1):
  """Supervised training of an MPS classifier on a dataset.
  Args:
    mps: MatrixProductState classifier object.
    optimizer: TensorFlow optimizer object to use in training.
      A working option is AdamOptimizer with learning_rate=1e-4.
    x: Training data (encoded images) of shape (n_data, n_sites, d_phys)
    y: Training labels in one-hot format of shape (n_data, n_labels)
    x_val: Validation data to calculate loss and accuracy during training.
    y_val: Validation labels to calculate loss and accuracy during training.
    n_epochs: Total number of epochs to train.
    batch_size: Batch size for training.
    n_message: Every how many epoch to print messages (loss, accuracy, times).
  """
  data = tf.cast(x, dtype=mps.dtype)
  labels = tf.cast(y, dtype=mps.dtype)
  n_batch = len(x) // batch_size

  if x_val is not None:
    data_val = tf.cast(x_val, dtype=mps.dtype)
    labels_val = tf.cast(y_val, dtype=mps.dtype)
    n_batch_val = len(x_val) // batch_size

  history = {"loss": [], "acc": [], "total_time": [],
             "val_loss": [], "val_acc": []}

  start_time = time.time()
  for epoch in range(n_epochs):
    generator = ((data[i * batch_size: (i + 1) * batch_size],
                  labels[i * batch_size: (i + 1) * batch_size])
                 for i in range(n_batch))
    loss, logits = run_epoch(mps, generator, n_batch, optimizer)
    history["loss"].append(loss / len(x))
    history["acc"].append(
        (logits.numpy().argmax(axis=1) == y.argmax(axis=1)).mean())
    history["total_time"].append(time.time() - start_time)

    if x_val is not None:
      val_generator = ((data_val[i * batch_size: (i + 1) * batch_size],
                        labels_val[i * batch_size: (i + 1) * batch_size])
                       for i in range(n_batch_val))
      val_loss, val_logits = run_epoch(mps, val_generator, n_batch_val)
      history["val_loss"].append(val_loss / len(x_val))
      history["val_acc"].append(
          (val_logits.numpy().argmax(axis=1) == y_val.argmax(axis=1)).mean())

    if epoch % n_message == 0:
      print("\nEpoch: {}".format(epoch))
      print("Time: {}".format(history["total_time"][-1]))
      print("Loss: {}".format(history["loss"][-1]))
      print("Accuracy: {}".format(history["acc"][-1]))
      if x_val is not None:
        print("Validation Accuracy: {}".format(history["val_acc"][-1]))

  return mps, history
