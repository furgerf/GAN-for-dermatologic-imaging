#!/usr/bin/env python

import json
import os
import shutil
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from model import Model
from perceptual_scores import PerceptualScores
from utils import data_subdirs


class Config:
  # pylint: disable=attribute-defined-outside-init,no-member,too-many-instance-attributes

  def __init__(self, config_file, start_time):
    self.start_time = start_time
    assert config_file is None or os.path.exists(config_file), "If specified, the config file must exist"
    self._config_file = config_file
    self._parser = ArgumentParser()

  def _load_config_file(self):
    if not self._config_file:
      return {}

    with open(self._config_file, "r") as fh:
      return json.load(fh)

  def _load_args(self):
    self._parser.add_argument("--config-file", type=str,
        help="Name of the json file containing the config.")

    # main global parameters
    self._parser.add_argument("--eid", type=str, required=True,
        help="ID of the evaluation")
    self._parser.add_argument("--model-name", type=str,
        help="Name of the model to use")
    self._parser.add_argument("--alternate-eid", type=str,
        help="ID of the alternate (second) evaluation")
    self._parser.add_argument("--alternate-model-name", type=str,
        help="Name of an alternate (second) model to use")
    self._parser.add_argument("--batch-size", type=int, default=128,
        help="The number of samples in one batch")
    self._parser.add_argument("--epochs", type=int, default=5000,
        help="The number of epochs to train for")
    self._parser.add_argument("--cycle", type=bool, default=False,
        help="Specify as true if this should be a CycleGAN")
    self._parser.add_argument("--noise-dimensions", type=int, default=100,
        help="The number of dimensions of the input noise")
    self._parser.add_argument("--extractor-name", type=str, default="VGG19", choices=PerceptualScores.EXTRACTOR_NAMES,
        help="The name of the feature extractor to use")

    # extra training parameters
    self._parser.add_argument("--train-disc-on-previous-images", type=bool, default=False,
        help="Specify if the discriminator should also be trained on the previous generator images")
    self._parser.add_argument("--train-disc-on-extra-targets", type=bool, default=False,
        help="Specify if the discriminator should be trained on extra target samples")
    self._parser.add_argument("--extra-disc-step-real", type=bool, default=False,
        help="Specify if the discriminator should be trained twice on real data per generator epoch")
    self._parser.add_argument("--extra-disc-step-both", type=bool, default=False,
        help="Specify if the discriminator should be trained twice on real and generated data per generator epoch")
    self._parser.add_argument("--use-extra-first-inputs", type=bool, default=False,
        help="Specify if the generator's second inputs should be used multiple times with different first inputs. Note: this doubles data set size!")

    # task modifications
    self._parser.add_argument("--conditioned-discriminator", type=bool, default=False,
        help="Specify if the discriminator should discriminate the combination of generation input+output")
    self._parser.add_argument("--inpainting", type=bool, default=False,
        help="Specify if the generator should perform image inpainting")
    self._parser.add_argument("--real-image-noise-stdev", type=float, default=0.0,
        help="The stdev of the noise that should be added to the real images for the discriminator")

    # augmentation, test data
    self._parser.add_argument("--augmentation-flip-lr", type=bool, default=False,
        help="Specify as true if the data should be augmented by L-R-flipped images")
    self._parser.add_argument("--augmentation-flip-ud", type=bool, default=False,
        help="Specify as true if the data should be augmented by U-D-flipped images")
    self._parser.add_argument("--test-data-percentage", type=float, default=0.0,
        help="If no test data is available, use the specified percentage of training data as a test set")
    self._parser.add_argument("--online-augmentation", type=bool, default=False,
        help="Specify as true if the data should be augmented on the fly - slower and less shuffled")

    # final experiments
    self._parser.add_argument("--final-experiment", type=bool, default=False,
        help="Specify as true if this is a final experiment")
    self._parser.add_argument("--keep-all-checkpoints", type=bool, default=False,
        help="Specify as true if all checkpoints should be kept")
    self._parser.add_argument("--keep-final-checkpoints", type=bool, default=False,
        help="Specify as true if checkpoints of the last 5 epochs should be kept")
    self._parser.add_argument("--scores-every-epoch", type=bool, default=False,
        help="Specify as true if the full scores should be calculated every epoch")

    directories = self._parser.add_argument_group("Directories")
    directories.add_argument("--data-dir", type=str,
        help="Directory containing the data set")
    directories.add_argument("--second-data-dir", type=str,
        help="Directory containing the second input data set, if different from the main input data dir")
    directories.add_argument("--target-data-dir", type=str,
        help="Directory containing the targer data set, if different from the input data dir")

    types = self._parser.add_argument_group("Input/output types")
    types.add_argument("--input-type", type=str, default="noise", choices=list(data_subdirs.keys()) + ["noise"],
        help="The type of the input for the generation")
    types.add_argument("--second-input-type", type=str, choices=list(data_subdirs.keys()) + ["noise"],
        help="The type of the secondary input for the generation")
    types.add_argument("--target-type", type=str, default="image", choices=data_subdirs.keys(),
        help="The type of the target of the generation")
    types.add_argument("--match-pattern", type=str, default=None,
        help="Pattern for files to match")

    losses = self._parser.add_argument_group("Loss weights")
    for loss in Model.all_individual_losses:
      losses.add_argument("--loss-{}".format(loss.replace("_", "-")), type=int, default=0,
          help="The weight of the {} loss for the generator.".format(loss))
      losses.add_argument("--loss-{}-power".format(loss.replace("_", "-")), type=int, default=1,
          help="The power of the {} loss for the generator.".format(loss))

    return vars(self._parser.parse_args())

  def _add_file_config_to_args(self, config_args, config_file):
    # the parsed args are the primary container and are updated with values from the config file
    # if the parsed args aren't left at default
    for file_key, file_value in config_file.items():
      key = file_key.replace("-", "_")
      if key not in config_args:
        tf.logging.error("Option '{}' ({}) from config file isn't a valid argument!".format(file_key, key))
        continue

      value = config_args[key]
      if self._parser.get_default(key) != value:
        tf.logging.warning("Ignoring option '{}={}' from config file as it's configured from the args as '{}'".format(key, file_value, value))
        continue

      tf.logging.debug("Setting '{}={}' from config file".format(key, file_value))
      config_args[key] = file_value

  def _add_derived_entries(self):
    if self.final_experiment:
      tf.logging.error("This is a final experiment, setting flags accordingly!")
      for arg, value in zip(["keep_all_checkpoints", "keep_final_checkpoints", "scores_every_epoch"], [True, True, True]):
        tf.logging.warning("Setting {}={} (was: {})".format(arg, value, self.__dict__[arg]))
        self.__dict__[arg] = value

    self.has_colored_input = self.input_type == "image"
    self.has_colored_second_input = self.second_input_type == "image"
    self.has_colored_target = self.target_type == "image"
    self.has_noise_input = self.input_type == "noise"

    total_loss_weight = np.sum(np.abs([self.__dict__["loss_{}".format(loss)] for loss in Model.all_individual_losses]))
    if total_loss_weight == 0:
      self.loss_adversarial = 1
      self.total_loss_weight = 1
    else:
      self.total_loss_weight = float(total_loss_weight)
    if self.loss_identity:
      assert self.has_colored_input and self.has_colored_target, "both domains need same dimensions"
    assert self.loss_adversarial > 0, "you probably didn't want to disable the adversarial loss"

    # by default, the disc has one output class
    self.discriminator_classes = 1

    # set up files and directories
    test_data_dir = self.data_dir + "-TEST"
    self.test_data_dir = test_data_dir if os.path.exists(os.path.join("data", test_data_dir)) else None
    test_second_data_dir = (self.second_data_dir or "/dev/null") + "-TEST"
    self.test_second_data_dir = test_second_data_dir if os.path.exists(os.path.join("data", test_second_data_dir)) else None
    test_target_data_dir = (self.target_data_dir or "/dev/null") + "-TEST"
    self.test_target_data_dir = test_target_data_dir if os.path.exists(os.path.join("data", test_target_data_dir)) else None
    assert not (bool(self.test_data_dir) and bool(self.test_data_percentage)), \
        "Shouldn't use training data for testing when test data is available"
    self.output_dir = os.path.join("output", self.eid)
    self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
    if self.keep_final_checkpoints:
      self.final_checkpoint_dir = os.path.join(self.output_dir, "final-checkpoints")
    self.samples_dir = os.path.join(self.output_dir, "samples")
    self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
    self.figures_dir = os.path.join(self.output_dir, "figures")
    self.hq_figures_dir = os.path.join(self.output_dir, "figures-hq")
    self.gradients_dir = os.path.join(self.output_dir, "gradients")
    if not os.path.exists(self.figures_dir):
      os.makedirs(self.figures_dir)
    if not os.path.exists(self.hq_figures_dir):
      os.makedirs(self.hq_figures_dir)
    if not os.path.exists(self.gradients_dir):
      os.makedirs(self.gradients_dir)

  def _cleanup(self):
    if self._config_file:
      if self.eid == "foo":
        tf.logging.warning("Only copying the config file")
        shutil.copy(self._config_file, os.path.join(self.output_dir, self._config_file))
      else:
        shutil.move(self._config_file, os.path.join(self.output_dir, self._config_file))
    del self._config_file
    del self._parser

  def setup(self):
    # load config from args and file
    config_args = self._load_args()
    config_file = self._load_config_file()

    # combine and apply config
    self._add_file_config_to_args(config_args, config_file)
    self.__dict__.update(config_args)

    # prepare additional stuff
    self._add_derived_entries()
    self._cleanup()
