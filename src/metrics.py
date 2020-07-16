#!/usr/bin/env python

from time import time
from abc import ABC, abstractmethod


class Metrics(ABC):
  def __init__(self, epoch, n_low_level_fids):
    self.epoch = epoch
    self.start_time = time()
    self.n_low_level_fids = n_low_level_fids
    assert self.n_low_level_fids == 4, "currently only support that since get_column_names is static"

  @abstractmethod
  def add_losses(self, gen_losses, disc_loss):
    pass

  @abstractmethod
  def add_discriminations(self, disc_on_real, disc_on_generated):
    pass

  @abstractmethod
  def add_perceptual_scores(self, fid, mmd, clustering_high, clustering_low, low_level_fids, combined_fid):
    pass

  @abstractmethod
  def add_disc_on_training_test(self, disc_on_training_mean, disc_on_training_std, disc_on_test_mean, disc_on_test_std):
    pass

  @abstractmethod
  def get_row_data(self):
    pass
