#!/usr/bin/env python

from time import time

import numpy as np

from metrics import Metrics
from model import Model
from utils import flatten


class OneWayMetrics(Metrics):
  def __init__(self, epoch, n_low_level_fids):
    self.gen_loss = []
    self.gen_losses = {loss: [] for loss in Model.all_individual_losses}
    self.disc_loss = []

    self.disc_on_real = []
    self.disc_on_generated = []

    self.disc_on_training_mean = np.nan
    self.disc_on_training_std = np.nan
    self.disc_on_test_mean = np.nan
    self.disc_on_test_std = np.nan

    self.fid = np.nan
    self.mmd = np.nan
    self.clustering_high = np.nan
    self.clustering_low = np.nan
    self.low_level_fids = [np.nan] * n_low_level_fids
    self.combined_fid = np.nan

    super(OneWayMetrics, self).__init__(epoch, n_low_level_fids)

  @staticmethod
  def get_column_names():
    epoch_fields = ["epoch", "epoch_time"]
    loss_fields = ["gen_loss_mean", "gen_loss_std", "disc_loss_mean", "disc_loss_std"]
    individual_loss_fields = flatten([["gen_{}_loss_mean".format(loss), "gen_{}_loss_std".format(loss)]
      for loss in Model.all_individual_losses]) # order matters!
    discrimination_fields = [
        "disc_on_real_mean", "disc_on_real_std", "disc_on_generated_mean", "disc_on_generated_std",
        "disc_on_training_mean", "disc_on_training_std", "disc_on_test_mean", "disc_on_test_std",
        ]
    perceptual_fields = ["fid", "mmd", "clustering_high", "clustering_low"] + \
        ["low_level_fid_{}".format(i+1) for i in range(4)] + ["combined_fid"]
    return epoch_fields + loss_fields + individual_loss_fields + discrimination_fields + perceptual_fields

  def add_losses(self, gen_losses, disc_loss):
    self.gen_loss.append(sum(gen_losses.values()).numpy())
    for loss in gen_losses:
      self.gen_losses[loss].append(gen_losses[loss].numpy())
    self.disc_loss.append(disc_loss.numpy())

  def add_discriminations(self, disc_on_real, disc_on_generated):
    self.disc_on_real.extend(disc_on_real.numpy().reshape(-1))
    self.disc_on_generated.extend(disc_on_generated.numpy().reshape(-1))

  def add_perceptual_scores(self, fid, mmd, clustering_high, clustering_low, low_level_fids, combined_fid):
    self.fid = fid.numpy()
    self.mmd = mmd.numpy()
    self.clustering_high = clustering_high
    self.clustering_low = clustering_low
    for i in range(self.n_low_level_fids):
      self.low_level_fids[i] = low_level_fids[i].numpy()
    self.combined_fid = combined_fid.numpy()

  def add_disc_on_training_test(self, disc_on_training_mean, disc_on_training_std, disc_on_test_mean, disc_on_test_std):
    self.disc_on_training_mean = disc_on_training_mean
    self.disc_on_training_std = disc_on_training_std
    self.disc_on_test_mean = disc_on_test_mean
    self.disc_on_test_std = disc_on_test_std

  def get_row_data(self):
    raw_data = [self.gen_loss, self.disc_loss] + [self.gen_losses[loss] for loss in Model.all_individual_losses] + \
        [self.disc_on_real, self.disc_on_generated] # order matters (individual losses)
    processed_data = [item for sublist in \
        [[np.mean(values) if values else np.nan, np.std(values) if values else np.nan] for values in raw_data] \
        for item in sublist]
    disc_on_training_test = [
        self.disc_on_training_mean, self.disc_on_training_std, self.disc_on_test_mean, self.disc_on_test_std]
    perceptual_scores = [self.fid, self.mmd, self.clustering_high, self.clustering_low] + self.low_level_fids + [self.combined_fid]
    return [self.epoch, time()-self.start_time] + processed_data + disc_on_training_test + perceptual_scores
