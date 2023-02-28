# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to compute an IntegratedGradients SaliencyMask."""

import tensorflow as tf
import numpy as np
from base import GradientSaliency


class IntegratedGradients(GradientSaliency):
	"""A SaliencyMask class that implements the integrated gradients method.

	https://arxiv.org/abs/1703.01365
	"""

	def GetMask(self, x_value, feed_dict = {}, x_baseline = None, x_steps = 30):
		"""Returns a integrated gradients mask.

		Args:
		  x_value: input ndarray.
		  x_baseline: Baseline value used in integration. Defaults to 0.
		  x_steps: Number of integrated steps between baseline and x.
		"""
		if x_baseline is None:
			x_baseline = np.zeros_like(x_value)

		assert x_baseline.shape == x_value.shape

		x_diff = x_value - x_baseline

		total_gradients = np.zeros_like(x_value)

		for alpha in np.linspace(0, 0.8, x_steps):
			x_step = x_baseline + alpha * x_diff
			# stdev_spread = 0.25
			# nsamples = 5
			# stdev = stdev_spread * (np.max(x_step) - np.min(x_step))
			# smoothed_gradients = np.zeros_like(x_step)
			# for i in range(nsamples):
			# 	noise = np.random.normal(0, stdev, x_step.shape)
			# 	x_plus_noise = x_step + noise
			# 	grad = super(IntegratedGradients, self).GetMask(x_plus_noise, feed_dict)
			# 	smoothed_gradients += (grad * grad)
			# smoothed_gradients = smoothed_gradients / nsamples
			# print("alpha: %s, smoothed_gradients: %s" % (alpha, np.mean(smoothed_gradients)))
			# total_gradients += smoothed_gradients

			total_gradients += np.abs(super(IntegratedGradients, self).GetMask(x_step, feed_dict))

		return total_gradients * x_diff / x_steps
