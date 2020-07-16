#!/usr/bin/env python

import csv
import logging
import os
import time
import traceback
from argparse import ArgumentParser
import coloredlogs
from datetime import datetime
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
from fastai.vision import *
from sklearn.metrics import recall_score

def configure_logging():
  coloredlogs.install(level="INFO")
  coloredlogs.DEFAULT_LEVEL_STYLES = {
      "debug": {"color": "white", "bold": False},
      "info": {"color": "white", "bold": True},
      "warning": {"color": "yellow", "bold": True},
      "error": {"color": "red", "bold": True},
      "fatal": {"color": "magenta", "bold": True},
      }
  logger = logging.getLogger("isic")
  log_format = "%(asctime)s %(levelname)s %(message)s"
  formatter = coloredlogs.ColoredFormatter(log_format)

  for handler in logger.handlers:
    handler.setFormatter(formatter)
  logger.propagate = False

def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eid", type=str, required=True)

  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--data-dir", type=str, required=True)
  parser.add_argument("--cycle1", action="store_true")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--tune-lr", type=float, default=1e-9)

  return parser.parse_args()

def prepare_learner(args):
  transforms = get_transforms(flip_vert=True, # enable flips in both directions but disable everything else
      max_rotate=None, max_lighting=None, max_zoom=0, max_warp=None, p_affine=0, p_lighting=0)
  logging.warning("Loading data from {}".format(args.data_dir))
  data = ImageDataBunch.from_folder(os.path.join("data", args.data_dir),
      seed=42, ds_tfms=transforms, size=256, bs=2).normalize(imagenet_stats)

  logging.warning("Setting up model")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    learner = load_learner(".")
  learner.data = data
  if len(learner.metrics) == 2:
    logging.fatal("Modifying original model")
    del learner.metrics[1]
    learner.model[-1][-1] = nn.Linear(in_features=512, out_features=learner.data.c, bias=True).cuda()
  logging.debug(learner.summary())

  return learner

def prepare_results(recorder, epochs_offset=0):
  assert len(recorder.metrics_names) == 1, "more metrics aren't implemented"
  columns = ("epoch", recorder.metrics_names[0], "train_loss", "valid_loss")

  epochs = range(epochs_offset, epochs_offset+len(recorder.nb_batches))
  logging.info("Preparing training results of {} epochs with offset {}".format(len(epochs), epochs_offset))

  metrics = [m[0].item() for m in recorder.metrics]

  # aggregate mean loss per epoch
  train_loss = []
  offset = 0
  for batch_size in recorder.nb_batches:
    batch_losses = recorder.losses[offset:offset+batch_size]
    offset += batch_size
    train_loss.append(np.mean(batch_losses))

  valid_loss = recorder.val_losses

  return [columns] + list(zip(epochs, metrics, train_loss, valid_loss))

def export_results(results, args):
  if not results:
    logging.warning("No results to export")
    return
  metrics_file = os.path.join(args.eval_dir, "metrics.csv")
  logging.info("Exporting results to '{}'".format(metrics_file))
  with open(metrics_file, "w") as fh:
    writer = csv.writer(fh)
    for row in results:
      writer.writerow(row)

def export_learner(learner, args):
  learner_file = os.path.join(args.eval_dir, "model.pkl")
  logging.info("Exporting learner to '{}'".format(learner_file))
  learner.export(learner_file)

def test_learner(learner, file_name, args):
  predictions, labels, losses = learner.get_preds(with_loss=True)
  interpretation = ClassificationInterpretation(learner, predictions, labels, losses)
  _ = interpretation.plot_confusion_matrix(return_fig=True)
  plt.savefig(os.path.join(args.eval_dir, file_name))

  _ = interpretation.plot_confusion_matrix(return_fig=True, normalize=True)
  plt.savefig(os.path.join(args.eval_dir, "normalized-" + file_name))

  scores = recall_score(labels, np.argmax(predictions, axis=1), average=None)
  logging.error("Mean class recall: {:.3f}".format(np.mean(scores)))
  logging.info("Per-class recall: {}".format(", ".join(["{}: {:.3f}".format(c, a) for c, a in zip(learner.data.valid_ds.y.classes, scores)])))

def main():
  results = None
  try:
    args = parse_arguments()
    args.eval_dir = os.path.join("output", args.eid)
    logging.info("Args: {}".format(args))

    learner = prepare_learner(args)

    data_set_size = len(learner.data.train_ds)
    total_training = args.epochs * 23000
    new_epochs = (total_training // data_set_size + 9) // 10 * 10
    logging.fatal("Changing epochs from {} to {} to account for data set size {}".format(args.epochs, new_epochs, data_set_size))
    logging.fatal("Samples per class: {}".format([len(np.where(learner.data.train_ds.y.items == c)[0]) for c in range(learner.data.train_ds.y.c)]))
    args.epochs = new_epochs

    if args.lr:
      logging.warning("Training with lr={}".format(args.lr))
      if args.cycle1:
        learner.fit_one_cycle(args.epochs, max_lr=args.lr)
      else:
        learner.fit(args.epochs, lr=args.lr)

      _ = learner.recorder.plot_losses(return_fig=True)
      plt.savefig(os.path.join(args.eval_dir, "initial-loss.png"))

      _ = learner.recorder.plot_metrics(return_fig=True)
      plt.savefig(os.path.join(args.eval_dir, "initial-metrics.png"))

      results = prepare_results(learner.recorder)
      test_learner(learner, "initial-confusion-matrix.png", args)
    else:
      logging.warning("Not performing initial training!")

    if not args.tune_lr:
      logging.warning("Not fine-tuning model!")
      return

    # fine-tune
    if not False:
      logging.info("Unfreezing entire learner")
      learner.unfreeze()
      logging.debug(learner.summary())
    else:
      logging.fatal("NOT UNFREEZING")

    logging.warning("Fine-tuning with lr={}".format(args.tune_lr))
    if args.cycle1:
      learner.fit_one_cycle(args.epochs, max_lr=args.tune_lr)
    else:
      learner.fit(args.epochs, lr=args.tune_lr)

    _ = learner.recorder.plot_losses(return_fig=True)
    plt.savefig(os.path.join(args.eval_dir, "tuned-loss.png"))

    _ = learner.recorder.plot_metrics(return_fig=True)
    plt.savefig(os.path.join(args.eval_dir, "tuned-metrics.png"))

    results = prepare_results(learner.recorder) if results is None else \
        results + prepare_results(learner.recorder, len(results)-1)[1:]
    test_learner(learner, "tuned-confusion-matrix.png", args)
  finally:
    if results:
      export_results(results, args)
      export_learner(learner, args)

if __name__ == "__main__":
  START_TIME = time.time()
  np.random.seed(42)
  configure_logging()
  try:
    main()
  except Exception as ex:
    logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    logging.info("Finished eval after {:.1f}m".format((time.time() - START_TIME) / 60))
