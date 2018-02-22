## fast-style-transfer webcam script

This is a fork of [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) which has an additional script, `run_webcam.py` to apply style models live to a webcam stream. Go to the README of the original page for instructions on how to train your own models, apply them to images and movies, and all the original functionality of that repository.

### Installation

 - [CUDA](https://developer.nvidia.com/cuda-downloads) + [CuDNN](https://developer.nvidia.com/cudnn)
 - [TensorFlow](https://www.tensorflow.org/install/) GPU-enabled
 - [OpenCV](https://pypi.python.org/pypi/opencv-python) (this is tested on cv 2.4, not most recent, but presumably both work)


### Setting up models

Pre-trained models for [Picasso, Hokusai, Kandinsky, Liechtenstein, Wu Guanzhong, Ibrahim el-Salahi, and Google Maps](https://drive.google.com/open?id=0B3WXSfqxKDkFUFl3YllzS1ZqbkU).

At the top of the file `run_webcam.py`, there are paths to model files and style images in the variable list `models`. They are not included in the repo because of space. If you'd like to use the pre-trained models referred to up there, these models may be [downloaded from this shared folder](https://drive.google.com/open?id=0B3WXSfqxKDkFUFl3YllzS1ZqbkU). To train your own, refer to the [original documentation](https://github.com/lengstrom/fast-style-transfer).

### Usage

    python run_webcam.py --width 360 --disp_width 800 --disp_source 1 --horizontal 1

There are three arguments:

 - `width` refers to the width in pixels of the image being restyled (the webcam will be scaled down or up to this size).  
 - `disp_width` is the width in pixels of the image to be shown on the screen. The restyled image is resized to this after being generated. Having `disp_width` > `width` lets you run the model more quickly but generate a bigger image of lesser quality.
 - `disp_source` is whether or not to display the content image (webcam) and corresponding style image alongside the output image (1 by default, i.e. True)
 - `horizontal` is whether to concatenate content/style with output image horizontally (1, which is default) or vertically (0). Only relevant if disp_source=1

You can toggle between the different models by hitting the 'a' and 's' keys on your keyboard.


### Example

![stylized webcam](styles/stylenet_webcam.gif)


### Requirements
You will need the following to run the above:
- TensorFlow 0.11.0
- Python 2.7.9, Pillow 3.4.2, scipy 0.18.1, numpy 1.11.2
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run TF on a GPU (cuda, etc)
- ffmpeg 3.1.3 if you want to stylize video

### Citation
```
  @misc{engstrom2016faststyletransfer,
    author = {Logan Engstrom},
    title = {Fast Style Transfer},
    year = {2016},
    howpublished = {\url{https://github.com/lengstrom/fast-style-transfer/}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project could not have happened without the advice (and GPU access) given by [Anish Athalye](http://www.anishathalye.com/). 
  - The project also borrowed some code from Anish's [Neural Style](https://github.com/anishathalye/neural-style/)
- Some readme/docs formatting was borrowed from Justin Johnson's [Fast Neural Style](https://github.com/jcjohnson/fast-neural-style)
- The image of the Stata Center at the very beginning of the README was taken by [Juan Paulo](https://juanpaulo.me/)

### License
Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (email: engstrom at my university's domain dot edu). Free for research/noncommercial use, as long as proper attribution is given and this copyright notice is retained.
