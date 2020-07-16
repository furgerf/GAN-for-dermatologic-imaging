#!/usr/bin/env python

from time import time

import numpy as np

from metrics import Metrics
from model import Model
from utils import flatten


class TwoWayMetrics(Metrics):
  # pylint: disable=too-many-instance-attributes

  def __init__(self, epoch, n_low_level_fids):
    self.first_gen_loss = []
    self.first_gen_losses = {loss: [] for loss in Model.all_individual_losses}
    self.first_disc_loss = []

    self.first_disc_on_real = []
    self.first_disc_on_generated = []

    self.first_disc_on_training_mean = np.nan
    self.first_disc_on_training_std = np.nan
    self.first_disc_on_test_mean = np.nan
    self.first_disc_on_test_std = np.nan

    self.first_fid = np.nan
    self.first_mmd = np.nan
    self.first_clustering_high = np.nan
    self.first_clustering_low = np.nan
    self.first_low_level_fids = [np.nan] * n_low_level_fids
    self.first_combined_fid = np.nan

    self.second_gen_loss = []
    self.second_gen_losses = {loss: [] for loss in Model.all_individual_losses}
    self.second_disc_loss = []

    self.second_disc_on_real = []
    self.second_disc_on_generated = []

    self.second_disc_on_training_mean = np.nan
    self.second_disc_on_training_std = np.nan
    self.second_disc_on_test_mean = np.nan
    self.second_disc_on_test_std = np.nan

    self.second_fid = np.nan
    self.second_mmd = np.nan
    self.second_clustering_high = np.nan
    self.second_clustering_low = np.nan
    self.second_low_level_fids = [np.nan] * n_low_level_fids
    self.second_combined_fid = np.nan

    super(TwoWayMetrics, self).__init__(epoch, n_low_level_fids)


  @staticmethod
  def get_column_names():
    epoch_fields = ["epoch", "epoch_time"]
    loss_fields = ["first_gen_loss_mean", "first_gen_loss_std", "first_disc_loss_mean", "first_disc_loss_std",
        "second_gen_loss_mean", "second_gen_loss_std", "second_disc_loss_mean", "second_disc_loss_std"]
    individual_loss_fields = flatten([["first_gen_{}_loss_mean".format(loss), "first_gen_{}_loss_std".format(loss)]
      for loss in Model.all_individual_losses]) + \
          flatten([["second_gen_{}_loss_mean".format(loss), "second_gen_{}_loss_std".format(loss)]
            for loss in Model.all_individual_losses]) # order matters!
    discrimination_fields = [
        "first_disc_on_real_mean", "first_disc_on_real_std", "first_disc_on_generated_mean", "first_disc_on_generated_std",
        "second_disc_on_real_mean", "second_disc_on_real_std", "second_disc_on_generated_mean", "second_disc_on_generated_std"
            ]
    disc_overfitting_fields = [
        "first_disc_on_training_mean", "first_disc_on_training_std", "first_disc_on_test_mean", "first_disc_on_test_std",
        "second_disc_on_training_mean", "second_disc_on_training_std", "second_disc_on_test_mean", "second_disc_on_test_std"
        ]
    perceptual_fields = ["first_fid", "first_mmd", "first_clustering_high", "first_clustering_low"] + \
        ["first_low_level_fid_{}".format(i+1) for i in range(4)] + ["first_combined_fid"] + \
        ["second_fid", "second_mmd", "second_clustering_high", "second_clustering_low"] + \
        ["second_low_level_fid_{}".format(i+1) for i in range(4)] + ["second_combined_fid"]
    return epoch_fields + loss_fields + individual_loss_fields + discrimination_fields + disc_overfitting_fields + perceptual_fields

  def add_losses(self, gen_losses, disc_loss):
    first_gen_losses, second_gen_losses = gen_losses
    first_disc_loss, second_disc_loss = disc_loss

    self.first_gen_loss.append(sum(first_gen_losses.values()).numpy())
    for loss in first_gen_losses:
      self.first_gen_losses[loss].append(first_gen_losses[loss].numpy())
    self.first_disc_loss.append(first_disc_loss.numpy())

    self.second_gen_loss.append(sum(second_gen_losses.values()).numpy())
    for loss in second_gen_losses:
      self.second_gen_losses[loss].append(second_gen_losses[loss].numpy())
    self.second_disc_loss.append(second_disc_loss.numpy())

  def add_discriminations(self, disc_on_real, disc_on_generated):
    first_disc_on_real, second_disc_on_real = disc_on_real
    first_disc_on_generated, second_disc_on_generated = disc_on_generated

    self.first_disc_on_real.extend(first_disc_on_real.numpy().reshape(-1))
    self.first_disc_on_generated.extend(first_disc_on_generated.numpy().reshape(-1))

    self.second_disc_on_real.extend(second_disc_on_real.numpy().reshape(-1))
    self.second_disc_on_generated.extend(second_disc_on_generated.numpy().reshape(-1))

  def add_perceptual_scores(self, fid, mmd, clustering_high, clustering_low, low_level_fids, combined_fid):
    first_fid, second_fid = fid
    first_mmd, second_mmd = mmd
    first_clustering_high, second_clustering_high = clustering_high
    first_clustering_low, second_clustering_low = clustering_low
    first_low_level_fids, second_low_level_fids = low_level_fids
    first_combined_fid, second_combined_fid = combined_fid

    self.first_fid = first_fid.numpy()
    self.first_mmd = first_mmd.numpy()
    self.first_clustering_high = first_clustering_high
    self.first_clustering_low = first_clustering_low
    for i in range(self.n_low_level_fids):
      self.first_low_level_fids[i] = first_low_level_fids[i].numpy()
    self.first_combined_fid = first_combined_fid.numpy()

    self.second_fid = second_fid.numpy()
    self.second_mmd = second_mmd.numpy()
    self.second_clustering_high = second_clustering_high
    self.second_clustering_low = second_clustering_low
    for i in range(self.n_low_level_fids):
      self.second_low_level_fids[i] = second_low_level_fids[i].numpy()
    self.second_combined_fid = second_combined_fid.numpy()

  def add_disc_on_training_test(self, disc_on_training_mean, disc_on_training_std, disc_on_test_mean, disc_on_test_std):
    first_disc_on_training_mean, second_disc_on_training_mean = disc_on_training_mean
    first_disc_on_training_std, second_disc_on_training_std = disc_on_training_std
    first_disc_on_test_mean, second_disc_on_test_mean = disc_on_test_mean
    first_disc_on_test_std, second_disc_on_test_std = disc_on_test_std

    self.first_disc_on_training_mean = first_disc_on_training_mean
    self.first_disc_on_training_std = first_disc_on_training_std
    self.first_disc_on_test_mean = first_disc_on_test_mean
    self.first_disc_on_test_std = first_disc_on_test_std

    self.second_disc_on_training_mean = second_disc_on_training_mean
    self.second_disc_on_training_std = second_disc_on_training_std
    self.second_disc_on_test_mean = second_disc_on_test_mean
    self.second_disc_on_test_std = second_disc_on_test_std

  def get_row_data(self):
    raw_data = [self.first_gen_loss, self.first_disc_loss, self.second_gen_loss, self.second_disc_loss] + \
        [self.first_gen_losses[loss] for loss in Model.all_individual_losses] + \
        [self.second_gen_losses[loss] for loss in Model.all_individual_losses] + \
        [self.first_disc_on_real, self.first_disc_on_generated] + \
        [self.second_disc_on_real, self.second_disc_on_generated] # order matters (individual losses)
    processed_data = [item for sublist in \
        [[np.mean(values) if values else np.nan, np.std(values) if values else np.nan] for values in raw_data] \
        for item in sublist]
    disc_on_training_test = [
        self.first_disc_on_training_mean, self.first_disc_on_training_std, self.first_disc_on_test_mean, self.first_disc_on_test_std,
        self.second_disc_on_training_mean, self.second_disc_on_training_std, self.second_disc_on_test_mean, self.second_disc_on_test_std
        ]
    perceptual_scores = [
        self.first_fid, self.first_mmd, self.first_clustering_high, self.first_clustering_low] + \
            self.first_low_level_fids + [self.first_combined_fid,
        self.second_fid, self.second_mmd, self.second_clustering_high, self.second_clustering_low] + \
            self.second_low_level_fids + [self.second_combined_fid
            ]
    return [self.epoch, time()-self.start_time] + processed_data + disc_on_training_test + perceptual_scores
