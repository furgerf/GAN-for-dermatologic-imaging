#!/usr/bin/env python


class GeneratorLossArgs:
  def __init__(self, generated_images, inputs, targets=None, reconstructed_images=None, identity_images=None):
    self._generated_images = generated_images
    self._inputs = inputs
    self._targets = targets
    self._reconstructed_images = reconstructed_images
    self._identity_images = identity_images

  @property
  def generated_images(self):
    return self._generated_images

  @property
  def inputs(self):
    return self._inputs

  @property
  def targets(self):
    return self._targets

  @property
  def reconstructed_images(self):
    return self._reconstructed_images

  @property
  def identity_images(self):
    return self._identity_images
