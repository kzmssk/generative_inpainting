import argparse
from pathlib import Path
from PIL import Image
import logging
import numpy as np
import tensorflow as tf
import copy
from inpaint_model import InpaintCAModel
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid


class Processor(object):
    def __init__(self, ckpt, input_width=680, input_height=512, mask_width=50):
        self.input_width = input_width
        self.input_height = input_height
        self.mask_width = mask_width

        # initialize tf session
        self.sess = tf.InteractiveSession()

        # define graph to process image
        self.input_plh, self.output = self.init_graph()

        # restore weight
        self.restore(str(ckpt))

    def init_graph(self):
        # input is concatenated input and mask
        # float32, min=0, max=255
        input_shape = (1, self.input_height, self.input_width * 2, 3)
        input_plh = tf.placeholder(tf.float32, input_shape, "input")

        model = InpaintCAModel()
        output = model.build_server_graph(input_plh)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        return input_plh, output

    def restore(self, ckpt):
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            assign_ops.append(tf.assign(
                var,
                tf.contrib.framework.load_variable(ckpt, var.name)
            ))
        self.sess.run(assign_ops)

    def load_image(self, filename):
        """ load image file and convert it.
        args:
            filename (Path): path of input image file
        returns:
            np.array shape = (self.width, self.height, 3),
                    dtype=np.float32,
                    value range=[0., 255.]
        """
        x = Image.open(str(filename)).convert("RGB").resize(
            (self.input_width, self.input_height))
        return np.asarray(x, dtype=np.uint8)

    def process(self, x, shift=False):
        """ process matrix x.
        args:
            x (np.array): input data whose shape = [self.width, self.height, 3]
        returns:
            np.array uint8 image array [w, h, k]
        """
        mask, masked = self.make_mask(x.astype(np.float32), shift)
        input_arr = np.concatenate([masked, mask], axis=1)
        input_arr = np.expand_dims(input_arr, axis=0)
        feed_dict = {self.input_plh: input_arr}
        output_arr = self.sess.run(self.output, feed_dict)
        output_arr = output_arr[0][:, :, ::-1]
        result = OrderedDict(masked=masked.astype(np.uint8),
                             mask=mask.astype(np.uint8),
                             output=output_arr)
        return result

    def make_mask(self, x, shift):
        mask = np.zeros((self.input_height, self.input_width, 3),
                        dtype=np.float32)
        mask[:, :self.mask_width, :] = 255.0

        masked = copy.deepcopy(x)
        if shift:
            masked[:, self.mask_width:, :] = x[:, :-self.mask_width, :]
        masked[:, :self.mask_width, :] = 255.0
        return mask, masked


def show_result(result, bg_color="darkgray"):
    num_cols = len(result.keys())
    fig = plt.figure()
    fig.patch.set_facecolor(bg_color)
    gs = grid.GridSpec(1, num_cols)

    i = 0
    for key, val in result.items():
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(val)
        ax.set_title(key)
        ax.set_axis_off()
        i += 1
    plt.show()


def gen(args, logger=None):
    """ process input image by InpaintCAModel. """
    if logger is None:
        logger = logging.getLogger(__name__)

    # check args
    assert args.input_image.exists(), f"{args.input_image} does not exist"
    assert args.ckpt.parent.exists(), f"{args.ckpt.parent} does not exist"

    # initialize Processor
    processor = Processor(args.ckpt)

    # preprocess image
    x = processor.load_image(args.input_image)
    result = processor.process(x)

    # show result
    show_result(result)


def gen_many_steps(args, logger=None, bg_color="darkgray"):
    if logger is None:
        logger = logging.getLogger(__name__)

    # check args
    assert args.input_image.exists(), f"{args.input_image} does not exist"
    assert args.ckpt.parent.exists(), f"{args.ckpt.parent} does not exist"

    # initialize Processor
    processor = Processor(args.ckpt, mask_width=10)

    # preprocess image
    x = processor.load_image(args.input_image)

    steps = 5
    fig = plt.figure()
    fig.patch.set_facecolor(bg_color)
    gs = grid.GridSpec(1, steps + 1)

    def show_image(ax, image, title):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title)

    ax = fig.add_subplot(gs[0, 0])
    show_image(ax, x, "input")

    for step in range(steps):
        ax = fig.add_subplot(gs[0, step + 1])
        x = processor.process(x, shift=True)["output"]
        show_image(ax, x, f"step {step + 1}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=Path)
    parser.add_argument("ckpt", type=Path)
    args = parser.parse_args()
    # gen(args)
    gen_many_steps(args)
