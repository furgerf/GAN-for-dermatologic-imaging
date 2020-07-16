#!/usr/bin/env python

# pylint: disable=ungrouped-imports,unused-import,too-many-statements,wildcard-import,unused-wildcard-import,too-many-locals

import csv
import os
import pickle
import re
from argparse import Namespace
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm
from scipy.misc import imread, imresize, imsave
from tqdm import tqdm, trange

from perceptual_scores import PerceptualScores
from utils import *

tf.enable_eager_execution()
plt.ion()

def plot_gen_disc_loss_plot(epochs, gen_loss_mean, gen_loss_std, disc_loss_mean, disc_loss_std, epoch_for_hours):
  plt.grid()
  plt.xticks(np.arange(epochs.min()-1, epochs.max()+50, 50))
  plt.axhline(y=0, linewidth=2, c="k")
  plt.plot(epochs, gen_loss_mean, "-", c="b", linewidth=2, label="Generator loss")
  plt.fill_between(epochs, gen_loss_mean-gen_loss_std, gen_loss_mean+gen_loss_std,
      alpha=0.3, lw=0, color="b")
  plt.plot(epochs, disc_loss_mean, "-", c="g", linewidth=2, label="Discriminator loss")
  plt.fill_between(epochs, disc_loss_mean-disc_loss_std, disc_loss_mean+disc_loss_std,
      alpha=0.3, lw=0, color="g")
  plt.legend()
  annotate_epoch_hours(epoch_for_hours)

def plot_disc_predictions(epochs, disc_on_fake_mean, disc_on_fake_std, disc_on_real_mean, disc_on_real_std, epoch_for_hours):
  plt.grid()
  plt.xticks(np.arange(epochs.min()-1, epochs.max()+50, 50))
  plt.plot(epochs, disc_on_fake_mean, "-", c="b", linewidth=2, label="Discriminator on fake")
  plt.fill_between(epochs, disc_on_fake_mean-disc_on_fake_std, disc_on_fake_mean+disc_on_fake_std,
      alpha=0.3, lw=0, color="b")
  plt.plot(epochs, disc_on_real_mean, "-", c="g", linewidth=2, label="Discriminator on real")
  plt.fill_between(epochs, disc_on_real_mean-disc_on_real_std, disc_on_real_mean+disc_on_real_std,
      alpha=0.3, lw=0, color="g")
  plt.axhline(y=0, linewidth=2, c="k")
  plt.axhline(y=0.5, linewidth=2, ls="--", c="k")
  plt.axhline(y=1, linewidth=2, c="k")
  annotate_epoch_hours(epoch_for_hours)

def plot_segmentation_scores(epochs, seg_score_mean, seg_score_std):
  plt.grid()
  plt.xticks(np.arange(epochs.min()-1, epochs.max()+50, 50))
  plt.axhline(y=0, linewidth=2, c="k")
  plt.axhline(y=1, linewidth=2, c="k")
  plt.plot(epochs, seg_score_mean, "-", c="purple", linewidth=2, label="Segmentation score")
  plt.fill_between(epochs, seg_score_mean-seg_score_std, seg_score_mean+seg_score_std,
      alpha=0.3, lw=0, color="purple")
  plt.legend()

def annotate_epoch_hours(epoch_for_hours, axis=None):
  axis = axis or plt.gca()
  for hour, epoch in enumerate(epoch_for_hours):
    if hour % 2 == 0:
      continue
    axis.annotate("{:d}h".format(hour+1), (epoch, axis.get_ylim()[1]), xytext=(epoch, 0.9*axis.get_ylim()[1]))

def match_individual_loss_column(column):
  return re.match("(?:(first|second)_)?gen_([A-z]+)_loss_mean", column)

def visualize_individual_losses(metrics, eid, epoch_for_hours, plot_total, clip_individual_losses):
  has_empty_prefix = [not match.group(1) for match in [match_individual_loss_column(column) for column in list(metrics)] if match]
  if all(has_empty_prefix):
    visualize_individual_losses_no_prefix(metrics, eid, epoch_for_hours, plot_total, clip_individual_losses)
  else:
    visualize_individual_losses_with_prefix(metrics, eid, epoch_for_hours, plot_total, clip_individual_losses)

def visualize_individual_losses_no_prefix(metrics, eid, epoch_for_hours, plot_total, clip_individual_losses):
  epochs = metrics["epoch"]
  plt.figure()
  cmap = cm.get_cmap("Set1")
  colors = {
      "adversarial": cmap(0),
      "relevancy": cmap(1),
      "reconstruction": cmap(2),
      "ground_truth": cmap(3),
      "binary": cmap(4),
      "variation": cmap(7),
      "patho_reconstruction": cmap(8),
      "identity": cmap(9),
      "total": cmap(6)
      }

  plt.title("{}: Individual generator losses".format(eid))
  plt.grid()
  plt.xticks(np.arange(epochs.min()-1, epochs.max()+50, 50))
  plt.axhline(y=0, linewidth=2, c="k")

  loss_means = []
  loss_stds = []

  for match in [match for match in [match_individual_loss_column(column) for column in list(metrics)] if match]:
    loss_name = match.group(2)
    mean = metrics["gen_{}_loss_mean".format(loss_name)]
    std = metrics["gen_{}_loss_std".format(loss_name)]
    plt.plot(epochs, mean, "-", c=colors[loss_name], linewidth=2, label=loss_name)
    plt.fill_between(epochs, mean-std, mean+std, alpha=0.3, lw=0, color=colors[loss_name])
    loss_means.append(mean.values)
    loss_stds.append(std.values)

  if plot_total:
    mean = metrics["gen_loss_mean"]
    std = metrics["gen_loss_std"]
    plt.plot(epochs, mean, "-", c=colors["total"], linewidth=2, label="total")
    plt.fill_between(epochs, mean-std, mean+std, alpha=0.3, lw=0, color=colors["total"])
    loss_means.append(mean.values)
    loss_stds.append(std.values)

  if clip_individual_losses:
    means = np.array(loss_means)
    stds = np.array(loss_stds)
    limits = (min(0, np.percentile(means, 5)-np.percentile(stds, 95)), np.percentile(means, 95)+np.percentile(stds, 95))
    plt.gca().set_ylim(limits)
  plt.legend()
  annotate_epoch_hours(epoch_for_hours)

def visualize_individual_losses_with_prefix(metrics, eid, epoch_for_hours, plot_total, clip_individual_losses):
  epochs = metrics["epoch"]
  plt.figure()
  cmap = cm.get_cmap("Set1")
  colors = {
      "adversarial": cmap(0),
      "relevancy": cmap(1),
      "reconstruction": cmap(2),
      "ground_truth": cmap(3),
      "identity": cmap(9),
      "total": cmap(6)
      }

  first_axis = plt.subplot(2, 1, 1)
  plt.title("{}: Individual losses first generator".format(eid))
  plt.grid()
  plt.xticks(np.arange(epochs.min()-1, epochs.max()+50, 50))
  plt.axhline(y=0, linewidth=2, c="k")
  second_axis = plt.subplot(2, 1, 2, sharex=first_axis, sharey=first_axis)
  plt.title("{}: Individual losses second generator".format(eid))
  plt.grid()
  plt.xticks(np.arange(epochs.min()-1, epochs.max()+50, 50))
  plt.axhline(y=0, linewidth=2, c="k")

  loss_means = []
  loss_stds = []

  for i, match in enumerate([match for match in [match_individual_loss_column(column) for column in list(metrics)] \
      if match and match.group(1) == "first"]):
    loss_name = match.group(2)
    mean = metrics["first_gen_{}_loss_mean".format(loss_name)]
    std = metrics["first_gen_{}_loss_std".format(loss_name)]
    first_axis.plot(epochs, mean, "-", c=cmap(i), linewidth=2, label=loss_name)
    first_axis.fill_between(epochs, mean-std, mean+std, alpha=0.3, lw=0, color=cmap(i))
    loss_means.append(mean.values)
    loss_stds.append(std.values)

    mean = metrics["second_gen_{}_loss_mean".format(loss_name)]
    std = metrics["second_gen_{}_loss_std".format(loss_name)]
    second_axis.plot(epochs, mean, "-", c=cmap(i), linewidth=2, label=loss_name)
    second_axis.fill_between(epochs, mean-std, mean+std, alpha=0.3, lw=0, color=cmap(i))
    loss_means.append(mean.values)
    loss_stds.append(std.values)

  if plot_total:
    mean = metrics["gen_loss_mean"]
    std = metrics["gen_loss_std"]
    plt.plot(epochs, mean, "-", c=colors["total"], linewidth=2, label="total")
    plt.fill_between(epochs, mean-std, mean+std, alpha=0.3, lw=0, color=colors["total"])
    loss_means.append(mean.values)
    loss_stds.append(std.values)

  if clip_individual_losses:
    means = np.array(loss_means)
    stds = np.array(loss_stds)
    limits = (min(0, np.percentile(means, 5)-np.percentile(stds, 95)), np.percentile(means, 95)+np.percentile(stds, 95))
    first_axis.set_ylim(limits)
    annotate_epoch_hours(epoch_for_hours, first_axis)
    second_axis.set_ylim(limits)
  first_axis.legend()
  second_axis.legend()
  annotate_epoch_hours(epoch_for_hours, second_axis)

def visualize_metrics(eid, images_per_epoch=None, plot_individual_losses=False, plot_total=False,
    plot_training_time=True, plot_mmd_reconstruction=False, plot_additional_fids=False, maximum_epoch=None,
    clip_individual_losses=False, additional_scores=True, fid_window=None,
    mmd_factor=int(5e2), clustering_high_factor=1e-3, clustering_low_factor=1e-1, reconstruction_factor=int(1e4)):
  # pylint: disable=too-many-statements,too-many-branches,too-many-locals,too-many-arguments
  metrics_file_glob = "output/*{}*/metrics.csv"
  metrics_file = glob(metrics_file_glob.format(eid))
  if not metrics_file:
    metrics_file_glob = "old-output/*{}*/metrics.csv"
    metrics_file = glob(metrics_file_glob.format(eid))
    if not metrics_file:
      print("No file matches glob '{}'".format(metrics_file_glob.format(eid)))
      return None
  if len(metrics_file) > 1:
    print("Several files match:\n  {}".format("\n  ".join(metrics_file)))
  metrics_file = metrics_file[0]
  eid = metrics_file[metrics_file.index("/")+1:-len("/metrics.csv")]
  print("Using metrics file '{}'".format(metrics_file))
  metrics = pd.read_csv(metrics_file)
  if maximum_epoch:
    metrics = metrics[:maximum_epoch]
  epoch_for_hours = []
  if "epoch_time" in metrics:
    runtime_hours = metrics.epoch_time.cumsum() / 3600
    for i in range(1, len(metrics)):
      indexes = runtime_hours.index[runtime_hours > i]
      if not indexes.any():
        break
      if plot_training_time:
        epoch_for_hours.append(metrics.iloc[indexes[0]].epoch)
    total_training_time = metrics.epoch_time.sum()
    print("Total training time: {:.1f}h, avg time per epoch: {:.1f}m{}".format(total_training_time / 3600,
      total_training_time / len(metrics) / 60, ", avg time per image: {:.2f}s".format(
        total_training_time / len(metrics) / images_per_epoch) if images_per_epoch else ""))

  # TODO: properformula to determine the number of plots on the main figure
  number_of_plots = 4 if "first_gen_loss_mean" in metrics else 2
  current_plot_number = 1
  plot_prefixes = {
      "": "",
      "first_": "First ",
      "second_": "Second "
      }

  plt.figure()
  plt.tight_layout()

  for prefix in sorted(plot_prefixes):
    if "{}gen_loss_mean".format(prefix) not in metrics:
      continue
    if current_plot_number == 1:
      first_axis = plt.subplot(number_of_plots, 1, current_plot_number)
    else:
      _ = plt.subplot(number_of_plots, 1, current_plot_number, sharex=first_axis, sharey=first_axis)
    plt.title("{}: {}Generator/discriminator loss".format(eid, plot_prefixes[prefix]))
    plot_gen_disc_loss_plot(metrics["epoch"],
        metrics["{}gen_loss_mean".format(prefix)], metrics["{}gen_loss_std".format(prefix)],
        metrics["{}disc_loss_mean".format(prefix)], metrics["{}disc_loss_std".format(prefix)],
        epoch_for_hours)
    if additional_scores:
      quantile = 0.8
      fid_key = "{}fid".format(prefix)
      if fid_key in metrics:
        not_nan = metrics[fid_key].notnull()
        if any(not_nan):
          _ = plt.gca().twinx()
          fids = metrics[fid_key][not_nan]
          plt.plot(metrics["epoch"][not_nan], fids, "-", c="purple", linewidth=1, marker="o", label="FID (final layer)")
          plt.ylim(0, fids.quantile(quantile))
          if fid_window:
            def moving_average(interval, window_size):
              window = np.ones(int(window_size)) / float(window_size)
              return np.convolve(interval, window, "same")
            fit = moving_average(fids, fid_window)
            plt.plot(metrics["epoch"][not_nan], fit, "--", c="magenta", linewidth=2, label="FID window {}".format(fid_window))
          if plot_mmd_reconstruction:
            mmd_key = "{}mmd".format(prefix)
            mmd = metrics[mmd_key][not_nan] * mmd_factor
            plt.plot(metrics["epoch"][not_nan], mmd, "-", c="brown", linewidth=1, marker="o", label="MMD*"+str(mmd_factor))
            reconstruction_mean_key = "{}reconstruction_mean".format(prefix)
            reconstruction_std_key = "{}reconstruction_std".format(prefix)
            if reconstruction_mean_key in metrics and any(metrics[reconstruction_mean_key].notnull()):
              plt.plot(metrics["epoch"][not_nan], metrics[reconstruction_mean_key][not_nan] * reconstruction_factor,
                  "-", c="orange", linewidth=1, marker="o", label="Reconstruction loss*"+str(reconstruction_factor))
              plt.fill_between(metrics["epoch"][not_nan],
                  (metrics[reconstruction_mean_key][not_nan]-metrics[reconstruction_std_key][not_nan])*10,
                  (metrics[reconstruction_mean_key][not_nan]+metrics[reconstruction_std_key][not_nan])*10,
                  alpha=0.3, lw=0, color="orange")
            clustering_high_key = "{}clustering_high".format(prefix)
            clustering_low_key = "{}clustering_low".format(prefix)
            if clustering_high_key in metrics and any(metrics[clustering_high_key].notnull()):
              plt.plot(metrics["epoch"][not_nan], metrics[clustering_high_key][not_nan] * clustering_high_factor,
                  "-", c="magenta", linewidth=1, marker="o", label="High-dim clustering score*"+str(clustering_high_factor))
              plt.plot(metrics["epoch"][not_nan], metrics[clustering_low_key][not_nan] * clustering_low_factor,
                  "-", c="orange", linewidth=1, marker="o", label="Low-dim clustering score*"+str(clustering_low_factor))
          low_level_fid_key = "{}low_level_fid_1".format(prefix)
          if plot_additional_fids and low_level_fid_key in metrics:
            not_nan = metrics[low_level_fid_key].notnull()
            if any(not_nan):
              cmap = plt.get_cmap("Reds")
              num_blocks = 4
              for i in range(num_blocks):
                low_level_fid_key = "{}low_level_fid_{}".format(prefix, i+1)
                low_level_fids = metrics[low_level_fid_key][not_nan]
                plt.plot(metrics["epoch"][not_nan], low_level_fids, "-", c=cmap((i+1)/(num_blocks+1)),
                    linewidth=1, marker="o", label="FID (block {})".format(i+1))
                plt.ylim(0, max(low_level_fids.quantile(quantile), plt.gca().get_ylim()[1]))
            combined_fid_key = "{}combined_fid".format(prefix)
            if combined_fid_key in metrics:
              not_nan = metrics[combined_fid_key].notnull()
              if any(not_nan):
                combined_fids = metrics[combined_fid_key][not_nan]
                plt.plot(metrics["epoch"][not_nan], combined_fids, "-", c="red", linewidth=1, marker="o", label="FID (combined)")
                plt.ylim(0, max(combined_fids.quantile(quantile), plt.gca().get_ylim()[1]))
          plt.legend(loc="upper left")
    current_plot_number += 1

  for prefix in sorted(plot_prefixes):
    if "{}disc_on_real_mean".format(prefix) not in metrics:
      continue
    if current_plot_number == 1:
      first_axis = plt.subplot(number_of_plots, 1, current_plot_number)
    else:
      _ = plt.subplot(number_of_plots, 1, current_plot_number, sharex=first_axis)
    fake_generated = "fake" if "{}disc_on_fake_mean".format(prefix) in metrics else "generated"
    assert "{}disc_on_{}_mean".format(prefix, fake_generated) in metrics, \
        "'{}disc_on_{}_mean' isn't among the columns!".format(prefix, fake_generated)
    plt.title("{}: {}Discriminator predictions on fake/real images".format(eid, plot_prefixes[prefix]))
    plot_disc_predictions(metrics["epoch"],
        metrics["{}disc_on_{}_mean".format(prefix, fake_generated)], metrics["{}disc_on_{}_std".format(prefix, fake_generated)],
        metrics["{}disc_on_real_mean".format(prefix)], metrics["{}disc_on_real_std".format(prefix)],
        epoch_for_hours)
    if additional_scores:
      disc_training_mean_key = "{}disc_on_training_mean".format(prefix)
      if disc_training_mean_key in metrics:
        not_nan = metrics[disc_training_mean_key].notnull()
        if any(not_nan):
          disc_on_training_mean = metrics[disc_training_mean_key][not_nan]
          plt.plot(metrics["epoch"][not_nan], disc_on_training_mean, "-", c="purple", linewidth=1, marker="o", label="Discriminator on training")
          disc_on_test_mean_key = "{}disc_on_test_mean".format(prefix)
          disc_on_test_mean = metrics[disc_on_test_mean_key][not_nan]
          plt.plot(metrics["epoch"][not_nan], disc_on_test_mean, "-", c="magenta", linewidth=1, marker="o", label="Discriminator on test")
          plt.plot(metrics["epoch"][not_nan], disc_on_training_mean-disc_on_test_mean, "-", c="orange",
              linewidth=1, marker="o", label="Difference training-test")
    plt.legend()
    current_plot_number += 1

  for prefix in sorted(plot_prefixes):
    if "{}seg_score_mean".format(prefix) not in metrics:
      continue
    if current_plot_number == 1:
      first_axis = plt.subplot(number_of_plots, 1, current_plot_number)
    else:
      _ = plt.subplot(number_of_plots, 1, current_plot_number, sharex=first_axis)
    plt.title("{}: {}Segmentation score".format(eid, plot_prefixes[prefix]))
    plot_segmentation_scores(metrics["epoch"],
        metrics["{}seg_score_mean".format(prefix)], metrics["{}seg_score_std".format(prefix)])
    current_plot_number += 1

  if plot_individual_losses and list(filter(None, [match_individual_loss_column(column) for column in metrics])):
    visualize_individual_losses(metrics, eid, epoch_for_hours, plot_total, clip_individual_losses)

  return metrics

def analyze_clustering_in_high_dim(activations, clusters=None, plotting=False):
  from sklearn.cluster import KMeans
  from sklearn.decomposition.pca import PCA

  clusters = [clusters] if clusters else np.arange(2, 13)
  kmeans = []
  for cluster in clusters:
    tf.logging.info("Fitting k-means with {} clusters in high-dimensional space".format(cluster))
    km = KMeans(n_clusters=cluster)
    km.fit(activations)
    kmeans.append(km)

  if plotting and len(clusters) > 1:
    plt.figure()
    plt.title("Number of clusters vs loss in high-dimensional space")
    plt.plot(clusters, [km.inertia_ for km in kmeans], marker="o")

  tf.logging.info("Fitting PCA")
  pca = PCA(n_components=2)
  transformed = pca.fit_transform(activations)
  tf.logging.info("Explained variance: {} ({:.5f})".format(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_)))

  if plotting:
    tf.logging.info("Plotting")
    cmap = plt.get_cmap("tab10")
    for i, cluster in enumerate(clusters):
      plt.figure()
      plt.title("{} clusters from high-dimensional space".format(cluster))
      for j in range(cluster):
        indexes = np.where(kmeans[i].labels_ == j)
        plt.scatter(transformed[indexes][:, 0], transformed[indexes][:, 1], label=str(j), color=cmap(j))
        center = pca.transform([kmeans[i].cluster_centers_[j]])[0]
        plt.scatter([center[0]], [center[1]], s=500, alpha=0.5, color=cmap(j), marker="+")
        distances = np.array([np.linalg.norm(center-transformed[index]) for index in indexes[0]])
        plt.gca().add_patch(plt.Circle(center, distances.mean(), color=cmap(j), alpha=0.3))
        plt.gca().add_patch(plt.Circle(center, distances.mean()+distances.std(), color=cmap(j), alpha=0.3))
      plt.legend()

  return pca, kmeans

def analyze_clustering_in_low_dim(activations, clusters=None, plotting=False):
  from sklearn.cluster import KMeans
  from sklearn.decomposition.pca import PCA

  tf.logging.info("Fitting PCA")
  pca = PCA(n_components=2)
  transformed = pca.fit_transform(activations)
  tf.logging.info("Explained variance: {} ({:.5f})".format(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_)))

  clusters = [clusters] if clusters else np.arange(2, 9)
  kmeans = []
  for cluster in clusters:
    tf.logging.info("Fitting k-means with {} clusters in low-dimensional space".format(cluster))
    km = KMeans(n_clusters=cluster)
    km.fit(transformed)
    kmeans.append(km)

  if plotting and len(clusters) > 1:
    plt.figure()
    plt.title("Number of clusters vs loss in low-dimensional space")
    plt.plot(clusters, [km.inertia_ for km in kmeans], marker="o")

  if plotting:
    tf.logging.info("Plotting")
    cmap = plt.get_cmap("tab10")
    for i, cluster in enumerate(clusters):
      plt.figure()
      plt.title("{} clusters from low-dimensional space".format(cluster))
      for j in range(cluster):
        indexes = np.where(kmeans[i].labels_ == j)
        plt.scatter(transformed[indexes][:, 0], transformed[indexes][:, 1], label=str(j), color=cmap(j))
        center = kmeans[i].cluster_centers_[j]
        plt.scatter([center[0]], [center[1]], s=500, alpha=0.5, color=cmap(j), marker="+")
        distances = np.amin(kmeans[i].transform(transformed[indexes]), axis=1)
        plt.gca().add_patch(plt.Circle(center, distances.mean(), color=cmap(j), alpha=0.3))
        plt.gca().add_patch(plt.Circle(center, distances.mean()+distances.std(), color=cmap(j), alpha=0.3))
      plt.legend()

  return pca, kmeans

def compute_clustering_scores(args):
  # pylint: disable=protected-access

  tf.logging.fatal("Computing clustering scores")

  extractor = PerceptualScores(args)
  extractor.initialize()

  plotting = True
  runs = [
      ("high", *analyze_clustering_in_high_dim(extractor._real_activations.numpy(), 7, plotting)),
      ("low", *analyze_clustering_in_low_dim(extractor._real_activations.numpy(), 4, plotting)),
      ]

  for run in runs:
    dim, pca, kmeans = run
    kmeans = kmeans[0]
    # generate_samples(args, generator, len(load_image_names(args.data_dir)))

    all_image_names = [os.path.join(args.samples_dir, sample) for sample in sorted(os.listdir(args.samples_dir)) if sample.endswith(".png")]

    activations_file = os.path.join(args.samples_dir, "activations_{}.npy".format(args.extractor_name))
    if os.path.exists(activations_file):
      tf.logging.info("Loading activations from {}".format(activations_file))
      generated_activations = tf.convert_to_tensor(np.load(activations_file))
    else:
      tf.logging.warning("Computing activations for generated images in '{}'".format(args.samples_dir))
      generated_activations = extractor._get_activations_from_images(all_image_names)
      tf.logging.info("Saving activations to {}".format(activations_file))
      np.save(activations_file, generated_activations.numpy())

    if dim == "low":
      transformed = pca.transform(generated_activations)
      labels = kmeans.predict(transformed)
      if plotting:
        cmap = plt.get_cmap("tab10")
        for j in range(kmeans.n_clusters):
          indexes = np.where(labels == j)
          plt.scatter(transformed[indexes][:, 0], transformed[indexes][:, 1], label=str(j), color=cmap(j), marker="x")
    else:
      labels = kmeans.predict(generated_activations)
      transformed = pca.transform(generated_activations)
      if plotting:
        cmap = plt.get_cmap("tab10")
        for j in range(kmeans.n_clusters):
          indexes = np.where(labels == j)
          plt.scatter(transformed[indexes][:, 0], transformed[indexes][:, 1], label=str(j), color=cmap(j), marker="x")

  plt.show()

def compare_perceptual_scores():
  # pylint: disable=line-too-long,too-many-branches
  data_full = pd.read_csv(os.path.join("analysis-results", "ps-data-full.csv")).values
  data_factor_2 = pd.read_csv(os.path.join("analysis-results", "ps-data-factor-2.csv")).values
  data_factor_4 = pd.read_csv(os.path.join("analysis-results", "ps-data-factor-4.csv")).values
  data_mobile_net_clustering = pd.read_csv(os.path.join("analysis-results", "ps-data-mobile-net-clustering.csv")).values
  for i in range(len(data_mobile_net_clustering)):
    data_mobile_net_clustering[i, 0] = np.log(data_mobile_net_clustering[i, 0])

  data_all_scores = np.array([
    pd.read_csv(os.path.join("analysis-results", "ps-data-all-scores-0.csv")).values,
    pd.read_csv(os.path.join("analysis-results", "ps-data-all-scores-1.csv")).values,
    pd.read_csv(os.path.join("analysis-results", "ps-data-all-scores-2.csv")).values,
    pd.read_csv(os.path.join("analysis-results", "ps-data-all-scores-3.csv")).values
    ])

  evaluations = [
      "both-deep",
      "both-deep-fancy-small-filters",
      "both-deep-multiscale-disc-normal-dropout-fewer-filters-two-deep-disc",
      "both-deep-multiscale-disc-normal-dropout-fewer-filters-two-deep-disc-better-disc",
      "both-deep-multiscale-disc-normal-drouput",
      "both-deep-spatial-dropout-second-try",
      "both-deep-multiscale-disc-normal-dropout-fewer-filters-two-deep-disc-gen-too-large-images",
      ]
  extractors = [
      "MobileNetV2",
      "ResNet50",
      "VGG16",
      "VGG19"
      ]

  plt.figure()
  for j, data in enumerate([data_full, data_factor_4]):
    _ = plt.subplot(2, 2, 1+2*j)
    plt.title("FID")
    plt.xticks(range(len(extractors)), sorted(extractors))
    for i, evaluation in enumerate(evaluations):
      if i == 1:
        continue
      values = data[i]
      plt.plot(range(len(values)//2), values[:3], marker="o", label=evaluation)
    plt.legend()

    _ = plt.subplot(2, 2, 2+2*j)
    plt.title("k-MMD")
    plt.xticks(range(len(extractors)), sorted(extractors))
    for i, evaluation in enumerate(evaluations):
      if i == 1:
        continue
      values = data[i]
      plt.plot(range(len(values)//2), values[3:], marker="o", label=evaluation)
    plt.legend()

  plt.figure()
  _ = plt.subplot(1, 3, 1)
  plt.title("FID on full resolution")
  plt.xticks(range(len(extractors)), sorted(extractors))
  for i, evaluation in enumerate(evaluations):
    if i == 1:
      continue
    values = data_full[i]
    plt.plot(range(len(values)//2), values[:3], marker="o", label=evaluation)
  plt.legend()

  _ = plt.subplot(1, 3, 2)
  plt.title("FID on factor 2")
  plt.xticks(range(len(extractors)), sorted(extractors))
  for i, evaluation in enumerate(evaluations):
    if i == 1:
      continue
    values = data_factor_2[i]
    plt.plot(range(len(values)//2), values[:3], marker="o", label=evaluation)
  plt.legend()

  _ = plt.subplot(1, 3, 3)
  plt.title("FID on factor 4")
  plt.xticks(range(len(extractors)), sorted(extractors))
  for i, evaluation in enumerate(evaluations):
    if i == 1:
      continue
    values = data_factor_4[i]
    plt.plot(range(len(values)//2), values[:3], marker="o", label=evaluation)
  plt.legend()

  plt.figure()
  colors = plt.get_cmap("tab10")
  plt.title("Variants of clustering scores")
  x_labels = ["log(fid)", "high", "high-mean", "high-mean-std", "low", "low-mean", "low-mean-std", "high-mean-perc", "high-mean-std-perc", "high-mean-std-std-perc", "low-mean-perc", "low-mean-std-perc", "low-mean-std-std-perc"]
  plt.xticks(range(len(x_labels)), x_labels)
  for i, evaluation in enumerate(evaluations):
    if i == 1:
      continue
    color_index = 0 if i == 0 else i-1
    means = data_mobile_net_clustering[i, ::2]
    plt.plot(range(len(x_labels)), means, color=colors(color_index), label=evaluation, marker="o")
    # stds = data_mobile_net_clustering[i, 1::2]
    # plt.fill_between(range(len(x_labels)), means+stds, means-stds, color=colors(color_index), alpha=0.3, lw=0)
  plt.legend()

  plt.figure()
  factors = [1, 1e3, 1e-2, 1e-1]
  x_labels = [s.format(factors[i]) for i, s in enumerate(["fid*{}", "mmd*{}", "clustering-high*{}", "clustering-low*{}"])]
  for j, extractor in enumerate(extractors):
    plt.subplot(1, len(extractors), j+1)
    plt.title(extractor)
    plt.xticks(range(len(x_labels)), x_labels)
    for i, evaluation in enumerate(evaluations):
      if i == 1:
        continue
      plt.plot(range(len(x_labels)), data_all_scores[j, i]*factors, label=evaluation, marker="o")
    if j == 0:
      plt.legend()

def visualize_gradients(eid, plot_loss=False, plot_gradients=False, plot_magnitudes=True):
  evaluations = glob(os.path.join("output", "*" + eid + "*"))
  if not evaluations:
    evaluations = glob(os.path.join("old-output", "*" + eid + "*"))
    if not evaluations:
      print("No evaluation found for eid '{}'".format(eid))
      return None
  if len(evaluations) > 1:
    print("Several evaluations match:\n  {}".format("\n  ".join(evaluations)))
  print("Using gradients of evaluation '{}'".format(evaluations[0]))

  gradient_files = glob(os.path.join(evaluations[0], "gradients", "*"))
  all_gradients = {}
  print("Loading {} gradient files...".format(len(gradient_files)))
  for gradient_file in tqdm(gradient_files):
    epoch = int(re.match(r".*_at_epoch_(\d+).pkl", gradient_file).group(1).lstrip("0"))
    with open(gradient_file, "rb") as fh:
      all_gradients[epoch] = pickle.load(fh)

  epochs = sorted(all_gradients.keys())
  gen_gradients = {epoch: dict(all_gradients[epoch][0]) for epoch in epochs}
  gen_variables = list(sorted({item for sublist in [epoch_gradients.keys() for epoch_gradients in gen_gradients.values()] for item in sublist}))
  gen_losses = [all_gradients[epoch][1] for epoch in epochs]
  disc_gradients = {epoch: dict(all_gradients[epoch][2]) for epoch in epochs}
  disc_variables = list(sorted({item for sublist in [epoch_gradients.keys() for epoch_gradients in disc_gradients.values()] for item in sublist}))
  disc_losses = [all_gradients[epoch][3] for epoch in epochs]

  def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    # not that good but better than the others.. https://stackoverflow.com/a/41259922
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
      axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])

  print("Plotting {} of gradients".format("mean magnitude" if plot_magnitudes else "mean/std"))
  plt.figure()
  # plt.title("{}: Generator gradients".format(gradient_files[0].split("/")[1]))
  main_axis = plt.gca()
  plt.grid()
  cmap = cm.get_cmap("tab20")
  extra_cmap = cm.get_cmap("Spectral")
  main_axis.set_xlabel("Epoch")
  main_axis.set_xticks(epochs)
  main_axis.set_ylabel("Mean gradient magnitude per layer")
  def get_color(i):
    if i == 0:
      return "k"
    if i < 11:
      # always have the same fixed colors for the first layers
      return cmap(i-1)

    if i == 10:
      return "k"
    if i < 10:
      # always have the same fixed colors for the first layers
      return cmap(i)

    # spread the remaining colors evenly
    total = len(gen_variables) - 11 # 1 inclusive
    return extra_cmap((i-10.0)/total) if total else cmap(0.5)

  for color_index, i in enumerate([10, 8, 0, 6, 1, 5, 2, 9, 3, 7, 4]):
    variable = gen_variables[i]
  # for i, variable in enumerate(sorted(gen_variables)):
    variable_gradients = [gen_gradients[epoch][variable] if variable in gen_gradients[epoch] else np.array([np.nan]) for epoch in epochs]
    gradient_sizes = {np.prod(gradient.shape) for gradient in variable_gradients}
    if "dense" in variable:
      label = "Dense"
    elif "conv2d_transpose" in variable:
      label = "Deconv stage {}".format(int(-np.log2(int(variable.split("_")[2]))+10))
    else:
      stage = int(variable.split("/")[0].split("_")[1])+1 if "_" in variable else 1
      label = "Final conv stage {}".format(stage)
    # label = "{} ({})".format(label, format_human(max(gradient_sizes), 1))
    if plot_magnitudes:
      gradients_mean = np.array([np.abs(gradients).mean() for gradients in variable_gradients])
      main_axis.plot(epochs, gradients_mean, label=label, color=get_color(color_index))
    else:
      gradients_mean = np.array([gradients.mean() for gradients in variable_gradients])
      gradients_std = np.array([gradients.std() for gradients in variable_gradients])
      main_axis.plot(epochs, gradients_mean, color=cmap(i), linestyle="--", alpha=0.5)
      main_axis.plot(epochs, gradients_std, label=label, color=get_color(color_index))
  plt.yscale("log")
  main_axis.legend(loc=2)

  if plot_loss:
    loss_axis = main_axis.twinx()
    loss_axis.plot(epochs, gen_losses, lw=3, label="G loss")
    align_yaxis(main_axis, loss_axis)
    loss_axis.legend()

  # # plt.figure()
  # plt.title("{}: Discriminator gradients".format(gradient_files[0].split("/")[1]))
  # # main_axis = plt.gca()
  # # plt.grid()
  # # cmap = cm.get_cmap("tab10")
  # # main_axis.set_xticks(epochs)
  # for color_index, i in enumerate([8, 0, 6, 1, 5, 2, 9, 3, 7, 4, 10]):
  #   variable = disc_variables[i]
  #   variable_gradients = [disc_gradients[epoch][variable] if variable in disc_gradients[epoch] else np.array([np.nan]) for epoch in epochs]
  #   gradient_sizes = {np.prod(gradient.shape) for gradient in variable_gradients}
  #   if "dense" in variable:
  #     label = "Dense"
  #   elif "plain_conv" in variable:
  #     label = "Main conv stage {}".format(int(-np.log2(int(variable.split("_")[2]))+10))
  #   else:
  #     stage = int(variable.split("/")[0].split("_")[1])+1-10
  #     label = "Initial conv stage {}".format(stage)
  #   # label = "{} ({})".format(label, format_human(max(gradient_sizes), 1))
  #   if plot_magnitudes:
  #     gradients_mean = np.array([np.abs(gradients).mean() for gradients in variable_gradients])
  #     main_axis.plot(epochs, gradients_mean, label=label, color=get_color(color_index))
  #   else:
  #     gradients_mean = np.array([gradients.mean() for gradients in variable_gradients])
  #     gradients_std = np.array([gradients.std() for gradients in variable_gradients])
  #     main_axis.plot(epochs, gradients_mean, color=cmap(i), linestyle="--", alpha=0.5)
  #     main_axis.plot(epochs, gradients_std, label=label, color=get_color(color_index))
  # plt.yscale("log")
  # main_axis.legend(loc=2)

  if plot_loss:
    loss_axis = main_axis.twinx()
    loss_axis.plot(epochs, disc_losses, lw=3, label="D loss")
    align_yaxis(main_axis, loss_axis)
    loss_axis.legend()

  if plot_gradients:
    gen_grads = all_gradients[epochs[-1]][0]
    all_grads = np.concatenate([gradient.reshape(-1) for variable, gradient in gen_grads])
    percentile = 2
    vmin = np.percentile(all_grads, percentile)
    vmax = np.percentile(all_grads, 100-percentile)
    print(all_grads.shape, all_grads.min(), all_grads.max(), vmin, vmax)

    for variable, gradient in gen_grads:
      print(variable, gradient.shape)
      if len(gradient.shape) != 4:
        print("Skipping...")
        continue

      plt.figure()
      plt.title("{} {:.3f}+-{:.3f}".format(variable, gradient.mean(), gradient.std()))
      plt.imshow(gradient.reshape(gradient.shape[0]*gradient.shape[2], -1), cmap="binary", vmin=vmin, vmax=vmax)
      plt.colorbar()

def plot_disc_overfitting(with_stdev=False, one_plot_per_figure=False, show_plot_number=None):
  df = pd.read_csv(os.path.join("analysis-results", "disc-predictions.csv"))

  evals = df.eid
  data = df[df.keys()[1:]].values
  eval_data = {eid: data[i] for i, eid in enumerate(evals)}

  skip_patterns = [
      r".*test-from-training-data.*",
      r".*G-(half|double)-filters.*",
      r".*truncated-dataset.*",
      ]

  ignore_group = r"^#.*"
  eval_groups = [
      r"256-full-hand-medium-patho.*back", r"256-full-hand-medium-patho-both",
      r"256-full-hand-any-patho-back", r"256-half-hand-medium-patho-back",

      r"512-half-hand-medium-patho-back", r"512-half-hand-medium-patho-both",
      r"512-full-hand-any-patho-back", r"512-half-hand-any-patho-back",

      r"480p", r"128-half-hand-no-patho-back", r"128-seg2img", None,

      r"128-full-hand-medium-patho-back-(?!.*res|.*final-dropout)", r"128-full-hand-medium-patho-back-.*-(res|final-dropout)-", None, None,

      r"isic-3-all", r"isic-3-AKIEC", r"isic-3-BCC", r"isic-3-BKL",

      r"isic-3-DF", r"isic-3-MEL", r"isic-3-NV", r"isic-3-VASC",

      r"isic19-AK", r"isic19-BCC", r"isic19-BKL", r"isic19-DF",

      r"isic19-MEL", r"isic19-NV", r"isic19-SCC", r"isic19-VASC",

      r"G-128-removepatho", r"128-removepatho-.*(-all-|train-D-on-G-input)",
      r"128-removepatho-.*-(nopatho|patho-separate)-", r"128-removepatho-.*-inpainting-",

      r"128-removepatho-(.*D-4xS2-double|.*D-4xS2-quadruple)(?!.*-res-|.*inpainting|.*patho-separate).*-additional-targets_",
      r"128-removepatho-(.*D-4xS2-octuple|.*D-4xS2-sexdecuple)(?!.*-res-|.*inpainting|.*patho-separate)(?!.*-additional-targets-relevancy)",
      r"128-removepatho-(?!.*-res-|.*inpainting|.*patho-separate).*-additional-targets-relevancy",
      r"128-removepatho-.*-res-",

      r"128-addpatho-(.*D-4xS2-double|.*D-4xS2-quadruple)",
      r"128-addpatho-.*D-4xS2-octuple(?!.*steps|.*conditioned)",
      r"128-addpatho-.*D-4xS2-octuple.*(steps|conditioned)",
      None,

      r"final-480p", r"final-128", r"final-512", r"final-isic",

      ignore_group
      ]

  data_set_sizes = [
      "{} (more: {})".format(334, 923), 1488, 8270, 2488,
      472, 1546, 315, 5361,
      492, 85742, 2872, None,
      2872, 2872, None, None,
      10015, 327, 514, 1099,
      115, 1113, 6705, 142,
      867, 3323, 2624, 239,
      4522, 12875, 628, 253,
      2872, 2872, 2872, 2872,
      2872, 2872, 2872, 2872,
      2872, 2872, 2872, None,
      492, "2872/51023", "2498/472", None,
      ]

  x_labels = ["training", "minus 10%", "plus 10%", "lrflip", "udflip", "test", "training-test", "G FID"]
  metric_count = len(x_labels)

  cmap = cm.get_cmap("tab10")
  number_of_colors = 10
  line_styles = ["-", "--"]
  marker_styles = ["X", "x"]
  processed_evals = 0

  def find_last_fid_for_eval(eid, epoch):
    # TODO instead, maybe return mean of last 5 or so
    file_path = os.path.join("output", eid, "metrics.csv")
    if not os.path.exists(file_path):
      file_path = os.path.join("old-output", eid, "metrics.csv")
    return pd.read_csv(file_path).iloc[epoch-1].fid if os.path.exists(file_path) else np.nan

  skip = False
  for j, group_pattern in enumerate(eval_groups):
    if not group_pattern:
      continue
    if group_pattern == ignore_group:
      current_evals = [eid for eid in evals if re.match(group_pattern, eid)]
      processed_evals += len(current_evals)
      continue

    print("Group:", group_pattern)
    current_evals = [eid for eid in evals if re.match(group_pattern, eid)]
    processed_evals += len(current_evals)

    if one_plot_per_figure or j % 4 == 0:
      if show_plot_number is not None and show_plot_number != j//4+1:
        skip = True
        continue
      skip = False
      plt.figure()

    if skip:
      continue

    if not one_plot_per_figure:
      plt.subplot(2, 2, (j%4)+1)

    plt.title("{} (samples: {})".format(group_pattern, data_set_sizes[j]))
    plt.xticks(range(len(x_labels)), x_labels)
    plt.ylim(0, 1)
    main_axis = plt.gca()
    fid_axis = plt.twinx()

    for i, eval_key in enumerate(current_evals):
      print("  Eval:", eval_key)
      if any([re.match(pattern, eval_key) for pattern in skip_patterns]):
        print("    ... skipping")
        continue
      epoch = int(eval_data[eval_key][0])
      fid = find_last_fid_for_eval(eval_key, epoch)
      data = eval_data[eval_key][1:]
      color = cmap(i % number_of_colors)
      main_axis.plot(range(metric_count), list(data[::2]) + [np.nan, np.nan],
          label="[{}]: {}".format(epoch, eval_key[:eval_key.index("_")]), color=color, ls=line_styles[i//number_of_colors])
      main_axis.scatter([metric_count-2], [data[0] - data[-2]], color=color, marker=marker_styles[i//number_of_colors])
      fid_axis.scatter([metric_count-1], [fid], color=color, marker=marker_styles[i//number_of_colors])

      if with_stdev:
        main_axis.fill_between(range(metric_count), data[::2]-data[1::2], data[::2]+data[1::2],
            alpha=0.1, lw=0, color=color)
    if one_plot_per_figure:
      main_axis.legend()
    else:
      main_axis.legend(prop={"size": 6})

  if processed_evals != len(eval_data):
    print("Processed", processed_evals, "but have", len(eval_data), "evals to process!")

  return df

def plot_perceptual_scores(one_plot_per_figure=False):
  df = pd.read_csv(os.path.join("analysis-results", "perceptual-scores.csv"))

  descriptions = df.description
  data = df[df.keys()[1:]].values
  eval_data = {description: data[i] for i, description in enumerate(descriptions)}

  x_labels = ["vs normal", "vs lrflip", "vs udflip", "vs test", "1st vs 2nd", "1st vs 2nd shuffled"]

  cmaps = [cm.get_cmap("tab20b"), cm.get_cmap("tab20c")]

  plt.figure()
  plt.title("FID: training set vs augmentations/test set")
  plt.xticks(range(len(x_labels)), x_labels)

  base_data_sets = {
      0: "256-full-hand-medium-patho-both",
      1: "512-full-hand-medium-patho-both",

      4: "128-full-hand-medium-patho-back",
      5: "256-full-hand-medium-patho-back",
      6: "512-full-hand-medium-patho-back",

      8: "256-full-hand-any-patho-back",
      9: "512-full-hand-any-patho-back",

      12: "256-full-hand-any-patho-back truncated",

      16: "128-full-hand-no-patho-back",

      20: "256-half-hand-medium-patho-back",
      21: "512-half-hand-medium-patho-back",

      24: "256-half-hand-any-patho-back",
      25: "512-half-hand-any-patho-back",

      28: "480p",
      }

  for color_index in sorted(base_data_sets):
    base_data_set = base_data_sets[color_index]
    keys = \
        ["{}: normal vs {}".format(base_data_set, other) for other in ["normal", "lrflip", "udflip"]] + \
        ["{}: training vs test".format(base_data_set)] + \
        ["{}: first vs second part of training{}".format(base_data_set, is_shuffled) for is_shuffled in ["", " shuffled"]]
    data = [eval_data[key][0] if key in eval_data else np.nan for key in keys]
    color = cmaps[color_index // 20](color_index % 20)
    plt.plot(range(len(data)), data, label=base_data_set, marker="o", ms=5, color=color)
  if one_plot_per_figure:
    plt.legend()
  else:
    plt.legend(prop={"size": 6})

  return df

def compute_perceptual_scores(description, data_dir, other_data_dir):
  args = Namespace(batch_size=16, extractor_name="VGG19", data_dir=data_dir,
      samples_dir=os.path.join("data", other_data_dir, "image"), target_data_dir=None)
  perceptual_scores = PerceptualScores(args)
  perceptual_scores.initialize()
  # scores = perceptual_scores.compare_base_activations()
  scores = perceptual_scores.compute_scores_from_samples()
  print("Perceptual scores: FID {:.3f}, MMD {:.3f}, high-dim clustering {:.1f}, low-dim clustering {:.1f}".format(*scores))

  scores_file = os.path.join("analysis-results", "perceptual-scores.csv")
  with open(scores_file, "a", buffering=1) as fh:
    writer = csv.writer(fh)
    if os.stat(scores_file).st_size == 0:
      writer.writerow(["description", "fid", "mmd", "clustering_high", "clustering_low"])
    writer.writerow([description] + [score.numpy() if isinstance(score, tf.Tensor) else score for score in scores])

# assumes that akiec/df/vasc/gen are loaded activations from .npz
# plt.figure()
# lab = list(sorted(akiec.files))
# for i, (data, label) in enumerate([(akiec, "AKIEC"), (df, "DF"), (vasc, "VASC"), (gen, "generated")]):
#   variabilities = [data[l] for l in lab]
#   variabilities = np.array([[v[:, j].std() for j in range(variabilities[0].shape[1])] for v in variabilities])
#   mean = np.array([variabilities[j].mean() for j in range(len(variabilities))])
#   std = np.array([variabilities[j].std() for j in range(len(variabilities))])
#   c = plt.get_cmap("Set1")(i)
#   plt.plot(range(len(lab)), mean, c=c, label=label)
#   plt.fill_between(range(len(lab)), mean+std, mean-std, color=c, alpha=0.3, lw=0)
# plt.xticks(range(len(lab)), lab)
# plt.legend()

def plot_truncation_threshold_selection_report():
  x = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "None"]
  fid = [111.44694, 94.48309, 74.96208, 69.49739, 69.54845, 70.34248, 74.11462, 73.79687, 74.18476]
  disc_std = [0.00707, 0.00622, 0.02826, 0.09610, 0.12664, 0.13619, 0.14419, 0.13668, 0.14484]
  disc_mean = [0.00225, 0.00251, 0.01114, 0.04109, 0.07515, 0.09192, 0.09301, 0.07670, 0.08215]
  plt.figure()
  f = plt.plot(range(len(x)), fid, label="FID", color="r", marker="o")
  plt.gca().set_xlabel("Truncation threshold")
  plt.xticks(range(len(x)), x)
  plt.gca().set_ylabel("FID")
  plt.twinx()
  d = plt.plot(range(len(x)), disc_mean, label="D prediction", color="b", marker="o")
  plots = f + d
  labels = [l.get_label() for l in plots]
  plt.ylim(0, 1)
  plt.fill_between(range(len(x)), np.array(disc_mean)-disc_std, np.array(disc_mean)+disc_std, color="b", alpha=0.3, lw=0)
  plt.legend(plots, labels)
  plt.gca().set_ylabel("Discriminator prediction")

def plot_hands_discriminator_overfitting_report():
  df = pd.read_csv(os.path.join("analysis-results", "disc-predictions.csv"))

  evals = df.eid
  data = df[df.keys()[1:]].values
  eval_data = {eid: data[i] for i, eid in enumerate(evals)}
  eval_key = "final-480p-lrflip-D-5xS2-FS4-G-5xS2S1-FS5-test-data_20190912_124623"

  x_labels = ["training", "darker", "brighter", "flip: left-right", "flip: up-down", "test"]
  metric_count = len(x_labels)

  cmap = cm.get_cmap("tab10")
  line_styles = ["-", "--"]
  marker_styles = ["X", "x"]

  plt.figure()
  plt.xticks(range(len(x_labels)), x_labels)
  plt.ylim(0, 1)

  epoch = int(eval_data[eval_key][0])
  data = eval_data[eval_key][1:]
  color = cmap(0)
  plt.plot(range(metric_count), list(data[::2]),
      label="[{}]: {}".format(epoch, eval_key[:eval_key.index("_")]), color=color, ls="-", marker="o")
  plt.fill_between(range(metric_count), data[::2]-data[1::2], data[::2]+data[1::2],
      alpha=0.3, lw=0, color=color)
  plt.xlabel("Input")
  plt.ylabel("Prediction")

def plot_classification_training(eid):
  metrics_file_glob = "classifications/*{}*/metrics.csv"
  metrics_file = glob(metrics_file_glob.format(eid))
  if not metrics_file:
    print("No file matches glob '{}'".format(metrics_file_glob.format(eid)))
    return None
  if len(metrics_file) > 1:
    print("Several files match:\n  {}".format("\n  ".join(metrics_file)))
  metrics_file = metrics_file[0]

  df = pd.read_csv(metrics_file)

  plt.figure()
  plt.title(metrics_file)
  plt.plot(df.epoch, df.accuracy, marker="o", label="Accuracy")
  plt.plot(df.epoch, df.train_loss, marker="o", label="Training loss")
  plt.plot(df.epoch, df.valid_loss, marker="o", label="Validation loss")
  plt.legend()
  plt.axhline(y=1, linewidth=2, ls="--", c="k")

  return df

def compare_classification_recall():
  data = []
  labels = None
  ansi_escape = re.compile(r'''
    \x1B    # ESC
    [@-_]   # 7-bit C1 Fe
    [0-?]*  # Parameter bytes
    [ -/]*  # Intermediate bytes
    [@-~]   # Final byte
    ''', re.VERBOSE) # https://stackoverflow.com/a/14693789/2135142
  for dir_name in sorted(os.listdir("classifications"), key=lambda item: item[::-1]):
    file_name = os.path.join("classifications", dir_name, "evaluation.log")
    if not os.path.exists(file_name):
      print("Skipping", file_name)
      continue
    print(dir_name)
    mean_recall = None
    class_recalls = None
    with open(file_name) as fh:
      for line in fh:
        line = ansi_escape.sub("", line) # remvoe escape sequences
        if "Mean class recall" in line:
          mean_recall = float(line.split(" ")[-1])
        if "Per-class recall" in line:
          class_recalls = [float(l.rstrip(",")) for l in line.split(" ")[8::2]]
          if not labels:
            labels = ["Mean"] + [l.rstrip(":") for l in line.split(" ")[7::2]]
    if mean_recall is None or class_recalls is None:
      print("Skipping", file_name)
      continue
    data.append((dir_name, mean_recall, class_recalls))

  mean_recalls = np.array([d[1] for d in data])
  mean_recall_index = mean_recalls.argsort()[-10:][::-1]
  print("\nTop mean recalls")
  for i, index in enumerate(mean_recall_index):
    print(i+1, data[index])

  x = range(len(labels))
  plt.figure()
  cmap = cm.get_cmap("tab20")
  skipped = 0
  for i, d in enumerate(data):
    name, mean, classes = d
    if ("tune" in name or "lr-1e-4" in name) and "cycle" not in name:
      plt.plot(x, [mean] + classes, marker="o", lw=0.5, ls="--", label=name, c=cmap(i-skipped))
    else:
      skipped += 1
      continue
      plt.plot(x, [mean] + classes, marker="o", lw=0.5, ls="--", alpha=0.2, label=name, c="k")
  plt.legend()
  plt.xticks(x, labels)
  plt.gca().set_ylim(0, 1)


def plot_perceptual_scores_report():
  df = pd.read_csv(os.path.join("analysis-results", "perceptual-scores.csv"))

  descriptions = df.description
  data = df[df.keys()[1:]].values
  eval_data = {description: data[i] for i, description in enumerate(descriptions)}

  x_labels = ["vs normal", "vs lrflip", "vs udflip", "vs lrflip udflip", "vs test", "1st vs 2nd", "1st vs 2nd shuffled"]

  cmap = cm.get_cmap("tab20")

  plt.figure()
  plt.title("FID: training set vs augmentations/test set")
  plt.xticks(range(len(x_labels)), x_labels)

  base_data_sets = {
      1: "r-128-full-hand-medium-patho-back",
      0: "r-128-full-hand-no-patho-back",

      3: "r-128-half-hand-medium-patho-back",
      2: "r-128-half-hand-no-patho-back",

      # 7: "r-512-full-hand-medium-patho-back",
      # 6: "r-512-full-hand-no-patho-back",

      5: "r-512-half-hand-medium-patho-back",
      4: "r-512-half-hand-no-patho-back",
      }
  data_set_counts = {
      "r-128-full-hand-medium-patho-back": 2872,
      "r-128-full-hand-no-patho-back": 51023,

      "r-128-half-hand-medium-patho-back": 9333,
      "r-128-half-hand-no-patho-back": 85742,

      "r-512-full-hand-medium-patho-back": 11,
      "r-512-full-hand-no-patho-back": 225,

      "r-512-half-hand-medium-patho-back": 472,
      "r-512-half-hand-no-patho-back": 2498
      }

  for color_index in sorted(base_data_sets):
    base_data_set = base_data_sets[color_index]
    label = "{} ({} samples)".format(base_data_set, data_set_counts[base_data_set])
    keys = \
        ["{}: normal vs {}".format(base_data_set, other) for other in ["normal", "lrflip", "udflip", "lrflip+udflip"]] + \
        ["{}: training vs test".format(base_data_set)] + \
        ["{}: first vs second part of training{}".format(base_data_set, is_shuffled) for is_shuffled in ["", " shuffled"]]
    data = [eval_data[key][0] if key in eval_data else np.nan for key in keys]
    color = cmap(color_index % 20)
    plt.plot(range(len(data)), data, label=label, marker="o", ms=5, color=color)
  plt.legend()

  return df



# In [89]: plt.figure()
#     ...: plt.xlim(0, 200)
#     ...: plt.ylim(-0.2, 0.6)
#     ...: plt.xlabel("Epochs")
#     ...: plt.ylabel("Difference discriminator on training and test data")
#     ...: ticks = ["{}/{}".format(i, 10*i) for i in range(0, 21, 2)]
#     ...: plt.xticks(np.array(range(len(ticks)))*20, ticks)
#     ...: plt.plot([0]+list(df_no["epoch"]*10), [0]+list((df_no["disc_on_training_mean"]-df_no["disc_on_test_mean"])), label="Healthy skin")
#     ...: plt.plot([0]+list(df_med["epoch"]), [0]+list(df_med["disc_on_training_mean"]-df_med["disc_on_test_mean"]), label="Skin with eczema")
#     ...: plt.axhline(y=0, linewidth=1, ls="--", c="k")
#     ...: plt.legend()


    # ...: plt.plot(df["epoch"], df["disc_on_training_mean"], color="b", label="Training data")
    # ...: plt.plot(df["epoch"], df["disc_on_test_mean"], color="r", label="Test data")
    # ...: plt.fill_between(df["epoch"], df["disc_on_training_mean"], df["disc_on_test_mean"], alpha=0.3, color="k")
    # ...: plt.legend()
    # ...: plt.xticks(range(0, len(df)+1, 5), range(0, len(df)+1, 5))
    # ...: plt.ylabel("Discriminator prediction")
    # ...: plt.xlabel("Epochs")
    # ...: plt.plot(df["epoch"], df["disc_on_training_mean"], color="b", label="Training data")
    # ...: plt.plot(df["epoch"], df["disc_on_test_mean"], color="r", label="Test data")
    # ...: plt.fill_between(df["epoch"], df["disc_on_training_mean"], df["disc_on_test_mean"], alpha=0.3, color="k")
    # ...: plt.legend()
    # ...: plt.xticks(range(0, len(df)+1, 5), range(0, len(df)+1, 5))
    # ...: plt.ylabel("Discriminator prediction")
    # ...: plt.xlabel("Epochs")


  # 12: labels = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
  # 13: counts = [867, 3323, 2624, 239, 4522, 12875, 628, 253]
  # 14: plt.plot(range(len(counts)), counts)
  # 15: plt.scatter(range(len(counts)), counts)
  # 16: plt.hist(range(len(counts)), counts)
  # 17: plt.hist(counts)
  # 18: plt.bar()
  # 19: plt.bar(1, 2)
  # 20: plt.bar(range(len(counts)), counts)
  # 21: plt.xticks(len(labels), labels)
  # 22: plt.xticks(range(len(labels)), labels)
  # 23: plt.xlabel("Lesion Class")
  # 24: plt.ylabel("Samples per class")
  # 25: plt.xlabel("Lesion class")
