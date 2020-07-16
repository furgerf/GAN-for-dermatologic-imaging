#!/usr/bin/env python

import csv
import os
import subprocess
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat

import numpy as np
import tensorflow as tf

from config import Config
from model import Model
from one_way_metrics import OneWayMetrics
from two_way_metrics import TwoWayMetrics
from utils import (configure_logging, format_human, load_class_from_module,
                   load_model)


def log_version():
  with open(os.devnull, "w") as dev_null:
    staged = subprocess.call(["git", "diff", "--staged", "--exit-code", "--name-only", "src"], stdout=dev_null)
    dirty = subprocess.call(["git", "diff", "--exit-code", "--name-only", "src"], stdout=dev_null)
    log_method = tf.logging.error if staged or dirty else tf.logging.info
    log_method("Current commit {}: '{}' ({}{}{})".format(
      subprocess.check_output(["git", "rev-list", "--count", "HEAD"]).strip().decode("utf-8"),
      subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).strip().decode("utf-8"),
      subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8"),
      " STAGED" if staged else "", " DIRTY" if dirty else ""))

def main():
  # handle arguments and config
  parser = ArgumentParser()
  parser.add_argument("--config-file", type=str,
      help="Name of the json file containing the config.")
  config = Config(parser.parse_known_args()[0].config_file, START_TIME)
  config.setup()

  # figure out which type of evaluation this is
  # pylint: disable=no-member
  if config.inpainting:
    assert config.second_input_type, "inpainting requires two inputs"
  if config.cycle:
    assert not config.has_noise_input, "don't use noise when doing cyclic generation"
    if config.second_input_type:
      evaluation_class = load_class_from_module("two_way_evaluations", "TwoWayTwoImagesToImageEvaluation")
    else:
      assert not config.second_data_dir, "don't specify second input when only using one"
      evaluation_class = load_class_from_module("two_way_evaluations", "TwoWayImageToImageEvaluation")
  else:
    if config.has_noise_input:
      assert not config.second_data_dir and not config.target_data_dir, "don't specify extra dirs with noise input"
      if config.alternate_eid:
        evaluation_class = load_class_from_module("one_way_evaluations", "NoiseImageSuperResolutionEvaluation")
      else:
        evaluation_class = load_class_from_module("one_way_evaluations", "NoiseToImageEvaluation")
    else:
      if config.second_input_type:
        if config.inpainting:
          evaluation_class = load_class_from_module("one_way_evaluations", "ImageInpaintingEvaluation")
        else:
          evaluation_class = load_class_from_module("one_way_evaluations", "TwoImagesToOneImageEvaluation")
      else:
        assert not config.second_data_dir, "don't specify second input when only using one"
        evaluation_class = load_class_from_module("one_way_evaluations", "ImageToImageEvaluation")

  model = load_model(config)
  evaluation = evaluation_class(model, config)

  tf.logging.fatal("Starting eval '{}' ({}) with model '{}', PID: {}".format(config.eid,
    evaluation.__class__.__name__, config.model_name, os.getpid()))
  log_version()
  tf.logging.warning("Args:\n{}".format(pformat(config.__dict__)))

  training_set_size, test_set_size = evaluation.load_data()
  evaluation.set_up_model()

  tf.logging.warning("Training for {} epochs of {} samples ({} total) - {} samples for testing".format(
    config.epochs, training_set_size, format_human(config.epochs * training_set_size), test_set_size))
  loss_info = ", ".join([info for info in [None if not config.__dict__["loss_{}".format(loss)] else
    "{}: ^{}*{}".format(loss, config.__dict__["loss_{}_power".format(loss)], config.__dict__["loss_{}".format(loss)])
    for loss in Model.all_individual_losses] if info])
  tf.logging.warning("Losses: {}".format(loss_info))
  metrics_file = os.path.join(config.output_dir, "metrics.csv")
  summary_writer = tf.contrib.summary.create_file_writer(config.tensorboard_dir)

  with open(metrics_file, "a", buffering=1) as fh, summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    metrics_writer = csv.writer(fh)
    if os.stat(metrics_file).st_size == 0:
      metrics_writer.writerow(TwoWayMetrics.get_column_names() if config.cycle else OneWayMetrics.get_column_names())
    evaluation.train(config.epochs, metrics_writer)

  with open(os.path.join("output", "finished-evals"), "a") as fh:
    fh.write("{}: {}\n".format(datetime.utcnow().replace(microsecond=0).isoformat(), config.eid))

if __name__ == "__main__":
  START_TIME = time.time()
  np.random.seed(42)
  tf.enable_eager_execution()
  configure_logging()
  try:
    main()
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished eval after {:.1f}m".format((time.time() - START_TIME) / 60))
