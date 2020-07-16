# Context

This code was developed during my master's course at Lucerne University of Applied Sciences and
Arts. The main part contains infrastructure to run different types of experiments around training
GANs to generate images with different conditions. Furthermore, there are additional scripts to
explore the trained models and analyze their learned representations.

It is also used for the paper "Applications of Generative Adversarial Networks to Dermatologic
Imaging" by Fabian Furger, Ludovic Amruthalingam, Alexander Navarini and Marc Pouly, published in
_Workshop on Artificial Neural Networks in Pattern Recognition_ (ANNPR), 2020.

# Usage

The framework is intended to be run inside a Docker container, where the specified python-packages
are installed. Individual experiments can then be run with the corresponding makefile targets or by
executing the relevant script in the `src/` directory directly.

While the framework and experiments were developed with a particular target domain - different types
of dermatology imaging - it can be applied to arbitrary image domains. Indeed, the same models have
also been trained on other images, without any code changes or reconfiguration.

