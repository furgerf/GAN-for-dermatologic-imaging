#!/usr/bin/env python

# pylint: disable=invalid-name,ungrouped-imports

import logging
import math
import os
from importlib import import_module

import coloredlogs
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import (array_ops, control_flow_ops, functional_ops,
                                   math_ops)


def get_hand_segmentation_for_image(image_file, hand_dir):
  return "{}/{}".format(hand_dir, os.path.basename(image_file).replace("image", "hand"))

def get_patho_segmentation_for_image(image_file, patho_dir):
  return "{}/{}".format(patho_dir, os.path.basename(image_file).replace("image", "patho"))

def get_combined_segmentation_for_image(image_file, combined_dir):
  return "{}/{}".format(combined_dir, os.path.basename(image_file).replace("image", "combined"))

image_subdir = "image"
hand_subdir = "hand"
patho_subdir = "patho"
combined_subdir = "combined"

data_subdirs = {
    image_subdir: image_subdir,
    hand_subdir: hand_subdir,
    patho_subdir: patho_subdir,
    combined_subdir: combined_subdir
    }
image_transformation_functions = {
    image_subdir: lambda x, y: x,
    hand_subdir: get_hand_segmentation_for_image,
    combined_subdir: get_combined_segmentation_for_image,
    patho_subdir: get_patho_segmentation_for_image
    }

def is_valid_file(file_name, pattern):
  return (not pattern or pattern in file_name) and (file_name.endswith(".png") or file_name.endswith(".jpg"))

def prepare_images(images, is_colored):
  tf.logging.info("Preparing {} images".format(len(images)))
  # normalize the images to the range of [-1, 1]
  normalized_images = np.array(images, dtype=np.float32) / 127.5 - 1
  return normalized_images if is_colored else \
    normalized_images.reshape(*normalized_images.shape, 1) # add dimension for "color depth"

def segmentation_score(output, ground_truth):
  assert output.shape[0] == ground_truth.shape[0]

  predicted = tf.cast(output >= 0, tf.uint8)
  actual = tf.cast(ground_truth >= 0, tf.uint8)

  tp = tf.count_nonzero(predicted * actual)
  # tn = tf.count_nonzero((predicted - 1) * (actual - 1))
  fp = tf.count_nonzero(predicted * (actual - 1))
  fn = tf.count_nonzero((predicted - 1) * actual)

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  return 2 * precision * recall / (precision + recall)

def logistic(logit):
  exp = np.exp(-logit) if isinstance(logit, np.ndarray) else tf.exp(-logit)
  return 1 / (1 + exp)



# since it's unvailable in 1.12.0, this is copied from:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
def kernel_classifier_distance_and_std_from_activations(real_activations,
                                                        generated_activations,
                                                        max_block_size=1024,
                                                        dtype=None):
  # pylint: disable=no-member
  """Kernel "classifier" distance for evaluating a generative model.
  This methods computes the kernel classifier distance from activations of
  real images and generated images. This can be used independently of the
  kernel_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like to precompute all of the
  activations before computing the classifier distance, or if we want to
  compute multiple metrics based on the same images. It also returns a rough
  estimate of the standard error of the estimator.
  This technique is described in detail in https://arxiv.org/abs/1801.01401.
  Given two distributions P and Q of activations, this function calculates
      E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
        - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]
  where k is the polynomial kernel
      k(x, y) = ( x^T y / dimension + 1 )^3.
  This captures how different the distributions of real and generated images'
  visual features are. Like the Frechet distance (and unlike the Inception
  score), this is a true distance and incorporates information about the
  target images. Unlike the Frechet score, this function computes an
  *unbiased* and asymptotically normal estimator, which makes comparing
  estimates across models much more intuitive.
  The estimator used takes time quadratic in max_block_size. Larger values of
  max_block_size will decrease the variance of the estimator but increase the
  computational cost. This differs slightly from the estimator used by the
  original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.
  The estimate of the standard error will also be more reliable when there are
  more blocks, i.e. when max_block_size is smaller.
  NOTE: the blocking code assumes that real_activations and
  generated_activations are both in random order. If either is sorted in a
  meaningful order, the estimator will behave poorly.
  Args:
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].
    max_block_size: integer, default 1024. The distance estimator splits samples
      into blocks for computational efficiency. Larger values are more
      computationally expensive but decrease the variance of the distance
      estimate. Having a smaller block size also gives a better estimate of the
      standard error.
    dtype: if not None, coerce activations to this dtype before computations.
  Returns:
   The Kernel Inception Distance. A floating-point scalar of the same type
     as the output of the activations.
   An estimate of the standard error of the distance estimator (a scalar of
     the same type).
  """

  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)
  real_activations.shape[1].assert_is_compatible_with(
      generated_activations.shape[1])

  if dtype is None:
    dtype = real_activations.dtype
    assert generated_activations.dtype == dtype
  else:
    real_activations = math_ops.cast(real_activations, dtype)
    generated_activations = math_ops.cast(generated_activations, dtype)

  # Figure out how to split the activations into blocks of approximately
  # equal size, with none larger than max_block_size.
  n_r = array_ops.shape(real_activations)[0]
  n_g = array_ops.shape(generated_activations)[0]

  n_bigger = math_ops.maximum(n_r, n_g)
  n_blocks = math_ops.to_int32(math_ops.ceil(n_bigger / max_block_size))

  v_r = n_r // n_blocks
  v_g = n_g // n_blocks

  n_plusone_r = n_r - v_r * n_blocks
  n_plusone_g = n_g - v_g * n_blocks

  sizes_r = array_ops.concat([
      array_ops.fill([n_blocks - n_plusone_r], v_r),
      array_ops.fill([n_plusone_r], v_r + 1),
  ], 0)
  sizes_g = array_ops.concat([
      array_ops.fill([n_blocks - n_plusone_g], v_g),
      array_ops.fill([n_plusone_g], v_g + 1),
  ], 0)

  zero = array_ops.zeros([1], dtype=dtypes.int32)
  inds_r = array_ops.concat([zero, math_ops.cumsum(sizes_r)], 0)
  inds_g = array_ops.concat([zero, math_ops.cumsum(sizes_g)], 0)

  dim = math_ops.cast(real_activations.shape[1], dtype)

  def compute_kid_block(i):
    'Compute the ith block of the KID estimate.'
    r_s = inds_r[i]
    r_e = inds_r[i + 1]
    r = real_activations[r_s:r_e]
    m = math_ops.cast(r_e - r_s, dtype)

    g_s = inds_g[i]
    g_e = inds_g[i + 1]
    g = generated_activations[g_s:g_e]
    n = math_ops.cast(g_e - g_s, dtype)

    k_rr = (math_ops.matmul(r, r, transpose_b=True) / dim + 1)**3
    k_rg = (math_ops.matmul(r, g, transpose_b=True) / dim + 1)**3
    k_gg = (math_ops.matmul(g, g, transpose_b=True) / dim + 1)**3
    return (-2 * math_ops.reduce_mean(k_rg) +
            (math_ops.reduce_sum(k_rr) - math_ops.trace(k_rr)) / (m * (m - 1)) +
            (math_ops.reduce_sum(k_gg) - math_ops.trace(k_gg)) / (n * (n - 1)))

  ests = functional_ops.map_fn(
      compute_kid_block, math_ops.range(n_blocks), dtype=dtype, back_prop=False)

  mn = math_ops.reduce_mean(ests)

  # nn_impl.moments doesn't use the Bessel correction, which we want here
  n_blocks_ = math_ops.cast(n_blocks, dtype)
  var = control_flow_ops.cond(
      math_ops.less_equal(n_blocks, 1),
      lambda: array_ops.constant(float('nan'), dtype=dtype),
      lambda: math_ops.reduce_sum(math_ops.square(ests - mn)) / (n_blocks_ - 1))

  return mn, math_ops.sqrt(var / n_blocks_)

def load_model(config):
  module_names = [
      "noise_to_image_models",
      "image_to_image_models",
      "deep_image_to_image_models",
      "deep_noise_to_image_models",
      "deep_noise_to_image_models",
      "deep_noise_to_square_image_models",
      "deep_image_super_resolution_models"
      ]
  for module_name in module_names:
    try:
      return load_class_from_module(module_name, config.model_name)(config)
    except AttributeError:
      pass
  assert False, "No model with name '{}' found".format(config.model_name)

def load_checkpoint(config, checkpoint_number=None, generator=None, discriminator=None,
    first_generator=None, second_generator=None, first_discriminator=None, second_discriminator=None):
  # pylint: disable=too-many-arguments
  tf.logging.info("Loading model from '{}', checkpoint {}".format(config.checkpoint_dir, checkpoint_number))
  models = {
      "generator": generator,
      "discriminator": discriminator,
      "first_generator": first_generator,
      "first_discriminator": first_discriminator,
      "second_generator": second_generator,
      "second_discriminator": second_discriminator
      }
  models = {key: models[key] for key in models if models[key]}
  checkpoint = tf.train.Checkpoint(**models)
  checkpoint_to_restore = "{}/ckpt-{}".format(config.checkpoint_dir, checkpoint_number) \
      if checkpoint_number else tf.train.latest_checkpoint(config.checkpoint_dir)
  checkpoint.restore(checkpoint_to_restore)

def load_image_names(data_dir, pattern=None):
  image_dir = os.path.join("data", data_dir, image_subdir)
  tf.logging.info("Loading image names from '{}'{}".format(
    image_dir, " matching pattern '{}'".format(pattern) if pattern else ""))
  return sorted([os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir) if is_valid_file(file_name, pattern)])

def augment_images(images, original, flip_lr, flip_ud):
  assert isinstance(images[0], (np.ndarray, tf.Tensor))
  if not flip_lr and not flip_ud:
    assert original
    return images

  augmented_images = []

  if flip_lr:
    tf.logging.info("Adding L-R-flipped images")
  if flip_ud:
    tf.logging.info("Adding U-D-flipped images")

  for image in images:
    if original:
      augmented_images.append(image)
    if flip_lr:
      augmented_images.append(np.fliplr(image))
    if flip_ud:
      augmented_images.append(np.flipud(image))
    if flip_lr and flip_ud:
      augmented_images.append(np.flipud(np.fliplr(image)))
  return augmented_images

def load_images(image_names, data_dir, image_type, original=True, flip_lr=False, flip_ud=False):
  image_dir = os.path.join("data", data_dir, data_subdirs[image_type])
  tf.logging.info("Loading {} images from '{}'".format(len(image_names), image_dir))
  is_colored = image_type == "image"
  get_file_name = lambda x: image_transformation_functions[image_type](x, image_dir)
  return prepare_images(
      augment_images(
        [imread(get_file_name(file_name), mode="RGB" if is_colored else "L") for file_name in image_names],
        original, flip_lr, flip_ud),
      is_colored)

def configure_logging():
  tf.logging.set_verbosity(tf.logging.INFO)
  coloredlogs.install(level="INFO")
  coloredlogs.DEFAULT_LEVEL_STYLES = {
      "debug": {"color": "white", "bold": False},
      "info": {"color": "white", "bold": True},
      "warning": {"color": "yellow", "bold": True},
      "error": {"color": "red", "bold": True},
      "fatal": {"color": "magenta", "bold": True},
      }
  logger = logging.getLogger("tensorflow")
  log_format = "%(asctime)s %(levelname)s %(message)s"
  formatter = coloredlogs.ColoredFormatter(log_format)

  for handler in logger.handlers:
    handler.setFormatter(formatter)
  logger.propagate = False

def get_memory_usage_string():
  used = tf.contrib.memory_stats.BytesInUse()
  total = tf.contrib.memory_stats.BytesLimit()
  peak = tf.contrib.memory_stats.MaxBytesInUse()
  return "{:.1f}/{:.1f}GB ({:.1f}%); peak: {:.1f}GB ({:.1f}%)".format(
      used/1e3**3, total/1e3**3, 100.0*used/total, peak/1e3**3, 100.0*peak/total)

def load_class_from_module(module_name, class_name):
  return getattr(import_module(module_name, class_name), class_name)

def flatten(list_of_lists):
  return [item for sublist in list_of_lists for item in sublist]

def format_human(number, digits=3):
  unit = 1000
  if number < unit:
    return str(number)
  magnitude = int(math.log(number) / math.log(unit))
  pre = "kMGTPE"[magnitude-1]
  scaled_number = number / math.pow(unit, magnitude)
  if scaled_number == int(scaled_number):
    scaled_number = int(scaled_number)
  else:
    scaled_number = round(scaled_number, digits)
  return "{}{}".format(scaled_number, pre)

def slerp(val, low, high):
  # https://github.com/dribnet/plat/blob/master/plat/interpolate.py
  """Spherical interpolation. val has a range of 0 to 1."""
  if val <= 0:
    return low
  if val >= 1:
    return high
  if np.allclose(low, high):
    return low
  omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
  so = np.sin(omega)
  return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def truncate_input(values, threshold):
  tf.logging.debug("Range before truncating: {} - {}".format(tf.reduce_min(values), tf.reduce_max(values)))
  def my_elementwise_func(x):
    if abs(x) < threshold:
      return x
    while abs(x) >= threshold:
      x = tf.random_normal((1,))[0]
    return x
  def recursive_map(inputs):
    if len(inputs.shape): # pylint: disable=len-as-condition
      return tf.map_fn(recursive_map, inputs)
    return my_elementwise_func(inputs)
  values = recursive_map(values)
  tf.logging.debug("Range after truncating: {} - {}".format(tf.reduce_min(values), tf.reduce_max(values)))
  return values
