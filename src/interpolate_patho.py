#!/usr/bin/env python

# pylint: disable=invalid-name

import os
import time
import traceback
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from tqdm import tqdm

from utils import (data_subdirs, load_checkpoint, load_image_names,
                   load_images, load_model, logistic)


def ndgrid(*args, **kwargs):
  """
  Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
  ``meshgrid`` for documentation).
  """
  kwargs['indexing'] = 'ij'
  return np.meshgrid(*args, **kwargs)

def bwperim(bw, n=4):
  """
  perim = bwperim(bw, n=4)
  Find the perimeter of objects in binary images.
  A pixel is part of an object perimeter if its value is one and there
  is at least one zero-valued pixel in its neighborhood.
  By default the neighborhood of a pixel is 4 nearest pixels, but
  if `n` is set to 8 the 8 nearest pixels will be considered.
  Parameters
  ----------
    bw : A black-and-white image
    n : Connectivity. Must be 4 or 8 (default: 8)
  Returns
  -------
    perim : A boolean image

  From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
  """

  if n not in (4, 8):
    raise ValueError('mahotas.bwperim: n must be 4 or 8')
  rows, cols = bw.shape

  # Translate image by one pixel in all directions
  north = np.zeros((rows, cols))
  south = np.zeros((rows, cols))
  west = np.zeros((rows, cols))
  east = np.zeros((rows, cols))

  north[:-1, :] = bw[1:, :]
  south[1:, :] = bw[:-1, :]
  west[:, :-1] = bw[:, 1:]
  east[:, 1:] = bw[:, :-1]
  idx = (north == bw) & (south == bw) & (west == bw) & (east == bw)
  if n == 8:
    north_east = np.zeros((rows, cols))
    north_west = np.zeros((rows, cols))
    south_east = np.zeros((rows, cols))
    south_west = np.zeros((rows, cols))
    north_east[:-1, 1:] = bw[1:, :-1]
    north_west[:-1, :-1] = bw[1:, 1:]
    south_east[1:, 1:] = bw[:-1, :-1]
    south_west[1:, :-1] = bw[:-1, 1:]
    idx &= (north_east == bw) & (south_east == bw) & (south_west == bw) & (north_west == bw)
  return ~idx * bw

def signed_bwdist(im):
  '''
  Find perim and return masked image (signed/reversed)
  '''
  im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
  return im

def bwdist(im):
  '''
  Find distance map of image
  '''
  dist_im = distance_transform_edt(1-im)
  return dist_im

def interp_shape(top, bottom, precision):
  # https://stackoverflow.com/questions/48818373/interpolate-between-two-images
  '''
  Interpolate between two contours

  Input: top
          [X,Y] - Image of top contour (mask)
          bottom
          [X,Y] - Image of bottom contour (mask)
          precision
            float  - % between the images to interpolate
              Ex: num=0.5 - Interpolate the middle image between top and bottom image
  Output: out
          [X,Y] - Interpolated image at num (%) between top and bottom

  '''
  if precision > 2:
    assert False, "Error: Precision must be between 0 and 1 (float)"

  if precision == 0.0:
    return top

  if precision == 1.0:
    return bottom

  top = signed_bwdist(top)
  bottom = signed_bwdist(bottom)

  # row,cols definition
  r, c = top.shape

  # Reverse % indexing
  precision = 1+precision

  # rejoin top, bottom into a single array of shape (2, r, c)
  top_and_bottom = np.stack((top, bottom))

  # create ndgrids
  points = (np.r_[0, 2], np.arange(r), np.arange(c))
  xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r**2, 2))
  xi = np.c_[np.full((r**2), precision), xi]

  # Interpolate for new plane
  out = interpn(points, top_and_bottom, xi)
  out = out.reshape((r, c))

  # Threshold distmap to values above 0 and convert to -1/1 binary map
  out = out > 0
  out = out.astype(np.float32) * 2 - 1

  return out


def parse_arguments():
  parser = ArgumentParser()

  parser.add_argument("--eval-dir", type=str, required=True,
      help="Directory of the evaluation to test (output)")
  parser.add_argument("--model-name", type=str, required=True,
      help="Name of the model to instantiate")
  parser.add_argument("--epoch", type=int, required=True,
      help="The epoch of the model to load")

  parser.add_argument("--data-dir", type=str,
      help="Directory containing the data set")
  parser.add_argument("--input-type", type=str, default="patho", choices=data_subdirs.keys(),
      help="The type of the input for the generation")
  parser.add_argument("--target-data-dir", type=str,
      help="Directory containing the targer data set, if different from the input data dir")
  parser.add_argument("--target-type", type=str, default="image", choices=data_subdirs.keys(),
      help="The type of the target of the generation")
  parser.add_argument("--conditioned-discriminator", type=bool, default=False,
      help="Specify if the discriminator should discriminate the combination of generation input+output")

  parser.add_argument("--description", type=str, default=None,
      help="An optional description of the images")
  parser.add_argument("--image-count", type=int, default=1,
      help="The number of images to generate")
  parser.add_argument("--interpolation-pairs", type=int, default=8,
      help="The number of pairs of noise vectors to interpolate")
  parser.add_argument("--interpolation-steps", type=int, default=8,
      help="The number of steps to interpolate")
  parser.add_argument("--width", type=int, default=3200,
      help="The width of the resulting image (multiple of 100)")
  parser.add_argument("--height", type=int, default=3200,
      help="The height of the resulting image (multiple of 100)")

  return parser.parse_args()

def main(start_time):
  tf.enable_eager_execution()

  # handle arguments and config
  args = parse_arguments()
  args.start_time = start_time

  tf.logging.info("Args: {}".format(args))

  assert args.input_type == "patho"
  assert args.target_type == "image"

  args.has_colored_input = args.input_type == "image"
  args.has_colored_target = args.target_type == "image"
  args.discriminator_classes = 1
  args.checkpoint_dir = os.path.join("output", args.eval_dir, "checkpoints")
  model = load_model(args)
  generator = model.get_generator()
  discriminator = model.get_discriminator()
  load_checkpoint(args, checkpoint_number=args.epoch//25, generator=generator, discriminator=discriminator)

  input_image_names = np.array(load_image_names(args.data_dir))
  # target_image_names = input_image_names if not args.target_data_dir else \
  #     np.array(load_image_names(args.target_data_dir))
  sample_indexes = np.random.choice(len(input_image_names), args.interpolation_pairs*2*args.image_count, replace=False)
  input_images = load_images(input_image_names[sample_indexes], args.data_dir, args.input_type)
  # target_images = load_images(target_image_names[sample_indexes], args.target_data_dir or args.data_dir, args.target_type)

  for image_number in range(args.image_count):
    tf.logging.info("Generating image {}/{}".format(image_number+1, args.image_count))
    plt.figure(figsize=(int(args.width/100), int(args.height/100)))

    with tqdm(total=args.interpolation_pairs*args.interpolation_steps) as pbar:
      for i in range(args.interpolation_pairs):
        index = 2*i+2*image_number*args.interpolation_pairs
        for j in range(args.interpolation_steps):
          pbar.update(1)
          input_sample = interp_shape(input_images[index, :, :, 0], input_images[index+1, :, :, 0], float(j) / (args.interpolation_steps-1))
          gen_input = tf.expand_dims(tf.expand_dims(input_sample, 0), -1)
          predictions = generator(gen_input, training=False)
          disc_input = tf.concat([gen_input, predictions], axis=-1) if args.conditioned_discriminator else predictions
          classifications = discriminator(disc_input, training=False)

          plt.subplot(args.interpolation_pairs*2, args.interpolation_steps, 2*i*args.interpolation_steps+j+1)
          plt.axis("off")
          if args.has_colored_input:
            plt.imshow(np.array((input_sample+1) * 127.5, dtype=np.uint8))
          else:
            plt.imshow(np.array((input_sample+1) * 127.5, dtype=np.uint8), cmap="gray", vmin=0, vmax=255)

          plt.subplot(args.interpolation_pairs*2, args.interpolation_steps, 2*i*args.interpolation_steps+j+1+args.interpolation_steps)
          plt.axis("off")
          plt.title(np.round(logistic(classifications[0]).numpy(), 5), fontsize=20)
          if args.has_colored_target:
            plt.imshow(np.array((predictions[0]+1) * 127.5, dtype=np.uint8))
          else:
            plt.imshow(np.array((predictions[0, :, :, 0]+1) * 127.5, dtype=np.uint8), cmap="gray", vmin=0, vmax=255)

    plt.tight_layout()
    figure_file = os.path.join("output", args.eval_dir, "noise_interpolation{}_{:03d}.png".format(
      "_{}".format(args.description) if args.description else "", image_number+1))
    plt.savefig(figure_file)
    plt.close()

  tf.logging.info("Finished generating {} images".format(args.image_count))


if __name__ == "__main__":
  START_TIME = time.time()
  np.random.seed(42)
  tf.logging.set_verbosity(tf.logging.INFO)
  try:
    main(START_TIME)
  except Exception as ex:
    tf.logging.fatal("Exception occurred: {}".format(traceback.format_exc()))
  finally:
    tf.logging.info("Finished eval after {:.1f}m".format((time.time() - START_TIME) / 60))
