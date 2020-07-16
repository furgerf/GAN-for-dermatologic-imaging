#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.misc import imread
from sklearn.cluster import KMeans
from sklearn.decomposition.pca import PCA
from tqdm import tqdm

from utils import (kernel_classifier_distance_and_std_from_activations,
                   load_image_names)


class PerceptualScores:
  EXTRACTOR_NAMES = ["MobileNetV2", "ResNet50", "VGG16", "VGG19"]

  def __init__(self, config):
    # pylint: disable=no-else-raise
    self._config = config
    self._real_activations = None
    if self._config.extractor_name == "MobileNetV2":
      from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
      from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
      model = MobileNetV2(include_top=False, weights="imagenet", alpha=1.4)
      self._preprocess = preprocess_input
      raise NotImplementedError("Need to update blocks...")
    elif self._config.extractor_name == "ResNet50":
      from tensorflow.keras.applications.resnet50 import ResNet50
      from tensorflow.keras.applications.resnet50 import preprocess_input
      model = ResNet50(include_top=False, weights="imagenet")
      self._preprocess = preprocess_input
      raise NotImplementedError("Need to update blocks...")
    elif self._config.extractor_name == "VGG16":
      from tensorflow.keras.applications.vgg16 import VGG16
      from tensorflow.keras.applications.vgg16 import preprocess_input
      model = VGG16(include_top=False, weights="imagenet")
      self._preprocess = preprocess_input
      raise NotImplementedError("Need to update blocks...")
    elif self._config.extractor_name == "VGG19":
      from tensorflow.keras.applications.vgg19 import VGG19
      from tensorflow.keras.applications.vgg19 import preprocess_input
      model = VGG19(include_top=False, weights="imagenet")
      self._extractor = Model(inputs=model.input, outputs=
          [model.get_layer("block{}_pool".format(i)).output for i in range(1, 6)])
      self._preprocess = preprocess_input
    else:
      raise ValueError("Unknown feature extractor '{}'".format(self._config.extractor_name))
    self._pca = None
    self._high_dimensional_kmeans = None
    self._low_dimensional_kmeans = None

  def _get_activations_from_images(self, all_image_names):
    activations = []
    data = tf.data.Dataset.from_tensor_slices(all_image_names).batch(self._config.batch_size)

    tf.logging.info("Computing activations for {} images".format(len(all_image_names)))
    for image_names in tqdm(data, total=len(all_image_names) // self._config.batch_size + 1):
      images = [imread(image_name.numpy().decode("utf-8"), mode="RGB") for image_name in image_names]
      batch = tf.cast(tf.stack(images), dtype=tf.float32)
      activations.append([tf.reduce_mean(features, axis=[1, 2]) for features in self._extractor(self._preprocess(batch))])
    return [tf.concat([act[i] for act in activations], axis=0) for i in range(len(activations[0]))]

  def _get_activations_from_generator(self, generator, data_set):
    activations = []
    tf.logging.debug("Computing activations for newly-generated samples")
    for batch in data_set:
      samples = tf.cast(tf.cast((generator(batch)+1) * 127.5, dtype=tf.int32), dtype=tf.float32) # denormalize to normal RGB
      activations.append([tf.reduce_mean(features, axis=[1, 2]) for features in self._extractor(self._preprocess(samples))])
    return [tf.concat([act[i] for act in activations], axis=0) for i in range(len(activations[0]))]

  def initialize(self, override_data_dir=None):
    assert self._real_activations is None

    data_dir = override_data_dir if override_data_dir else \
        (self._config.target_data_dir if self._config.target_data_dir else self._config.data_dir)
    activations_file = os.path.join("data", data_dir, "activations_{}.npz".format(self._config.extractor_name))
    if os.path.exists(activations_file):
      tf.logging.info("Loading activations from {}".format(activations_file))
      with np.load(activations_file) as activations:
        self._real_activations = [tf.convert_to_tensor(activations[f]) for f in sorted(activations.files)]
    else:
      tf.logging.warning("Computing activations for real images in '{}'".format(data_dir))
      self._real_activations = self._get_activations_from_images(load_image_names(data_dir))
      tf.logging.info("Saving activations to {}".format(activations_file))
      np.savez(activations_file, **{"block_{}".format(i): act.numpy() for i, act in enumerate(self._real_activations)})

    tf.logging.debug("Fitting PCA")
    self._pca = PCA(n_components=2)
    low_dimensional_real_activations = self._pca.fit_transform(self._real_activations[-1])
    tf.logging.debug("Explained variance: {} ({:.5f})".format(
      self._pca.explained_variance_ratio_, np.sum(self._pca.explained_variance_ratio_)))

    high_dimensional_clusters = 7
    tf.logging.debug("Clustering high-dimensional activations with {} clusters".format(high_dimensional_clusters))
    self._high_dimensional_kmeans = KMeans(n_clusters=high_dimensional_clusters)
    self._high_dimensional_kmeans.fit(self._real_activations[-1])
    tf.logging.debug("Inertia: {:.1f}".format(self._high_dimensional_kmeans.inertia_))

    low_dimensional_clusters = 4
    tf.logging.debug("Clustering low-dimensional activations with {} clusters".format(low_dimensional_clusters))
    self._low_dimensional_kmeans = KMeans(n_clusters=low_dimensional_clusters)
    self._low_dimensional_kmeans.fit(low_dimensional_real_activations)
    tf.logging.debug("Inertia: {:.1f}".format(self._low_dimensional_kmeans.inertia_))

  def _compute_scores_from_activations(self, generated_activations):
    fid = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(self._real_activations[-1], generated_activations[-1])
    mmd, _ = kernel_classifier_distance_and_std_from_activations(self._real_activations[-1], generated_activations[-1])
    low_level_fids = [
        tf.contrib.gan.eval.frechet_classifier_distance_from_activations(self._real_activations[i], generated_activations[i]) \
            for i in range(len(self._real_activations)-1)]
    combined_fid = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(
        tf.concat(self._real_activations, axis=-1), tf.concat(generated_activations, axis=-1))

    # high_dimensional_cluster_distances = tf.reduce_min(self._high_dimensional_kmeans.transform(generated_activations), axis=-1)
    # low_dimensional_cluster_distances = tf.reduce_min(self._low_dimensional_kmeans.transform(self._pca.transform(generated_activations)), axis=-1)
    # mean_std = lambda d: (tf.reduce_mean(d), tf.convert_to_tensor(np.std(d)))
    # return fid, k_mmd, mean_std(high_dimensional_cluster_distances), mean_std(low_dimensional_cluster_distances)

    return fid, mmd, -self._high_dimensional_kmeans.score(generated_activations[-1]), \
        -self._low_dimensional_kmeans.score(self._pca.transform(generated_activations[-1])), low_level_fids, combined_fid

  def compute_scores_from_samples(self):
    assert os.path.exists(self._config.samples_dir)
    all_image_names = [os.path.join(self._config.samples_dir, sample) for sample in \
        sorted(os.listdir(self._config.samples_dir)) if sample.endswith(".png")]

    activations_file = os.path.join(self._config.samples_dir, "activations_{}.npz".format(self._config.extractor_name))
    if os.path.exists(activations_file):
      tf.logging.info("Loading activations from {}".format(activations_file))
      generated_activations = tf.convert_to_tensor(np.load(activations_file))
    else:
      tf.logging.warning("Computing activations for generated images in '{}'".format(self._config.samples_dir))
      generated_activations = self._get_activations_from_images(all_image_names)
      tf.logging.info("Saving activations to {}".format(activations_file))
      np.savez(activations_file, **{"block_{}".format(i): act.numpy() for i, act in enumerate(self._real_activations)})

    tf.logging.info("Computing scores")
    return self._compute_scores_from_activations(generated_activations)

  def compute_scores_from_generator(self, generator, data_set):
    generated_activations = self._get_activations_from_generator(generator, data_set)

    tf.logging.debug("Computing scores")
    return self._compute_scores_from_activations(generated_activations)
