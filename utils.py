import scipy.misc
import numpy as np
import os
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist

class ImageData:

    def __init__(self, load_size, channels, custom_dataset):
        self.load_size = load_size
        self.channels = channels
        self.custom_dataset = custom_dataset


    @staticmethod
    def _crop(image, offset_height, offset_width, crop_height, crop_width):
        """Crops the given image using the provided offsets and sizes.
        Note that the method doesn't assume we know the input image size but it does
        assume we know the input image rank.
        Args:
          image: an image of shape [height, width, channels].
          offset_height: a scalar tensor indicating the height offset.
          offset_width: a scalar tensor indicating the width offset.
          crop_height: the height of the cropped image.
          crop_width: the width of the cropped image.
        Returns:
          the cropped (and resized) image.
        Raises:
          InvalidArgumentError: if the rank is not 3 or if the image dimensions are
            less than the crop size.
        """
        original_shape = tf.shape(image)

        rank_assertion = tf.Assert(
            tf.equal(tf.rank(image), 3),
            ['Rank of image must be equal to 3.'])
        with tf.control_dependencies([rank_assertion]):
            cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

        size_assertion = tf.Assert(
            tf.logical_and(
                tf.greater_equal(original_shape[0], crop_height),
                tf.greater_equal(original_shape[1], crop_width)),
            ['Crop size greater than the image size.'])

        offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

        # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
        # define the crop size.
        with tf.control_dependencies([size_assertion]):
            image = tf.slice(image, offsets, cropped_shape)
        return tf.reshape(image, cropped_shape)


    def _random_crop(self, image_list, crop_height, crop_width):
        """Crops the given list of images.
        The function applies the same crop to each image in the list. This can be
        effectively applied when there are multiple image inputs of the same
        dimension such as:
          image, depths, normals = _random_crop([image, depths, normals], 120, 150)
        Args:
          image_list: a list of image tensors of the same dimension but possibly
            varying channel.
          crop_height: the new height.
          crop_width: the new width.
        Returns:
          the image_list with cropped images.
        Raises:
          ValueError: if there are multiple image inputs provided with different size
            or the images are smaller than the crop dimensions.
        """
        if not image_list:
            raise ValueError('Empty image_list.')

        # Compute the rank assertions.
        rank_assertions = []
        for i in range(len(image_list)):
            image_rank = tf.rank(image_list[i])
            rank_assert = tf.Assert(
                tf.equal(image_rank, 3),
                ['Wrong rank for tensor  %s [expected] [actual]',
                 image_list[i].name, 3, image_rank])
            rank_assertions.append(rank_assert)

        with tf.control_dependencies([rank_assertions[0]]):
            image_shape = tf.shape(image_list[0])
        image_height = image_shape[0]
        image_width = image_shape[1]
        crop_size_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(image_height, crop_height),
                tf.greater_equal(image_width, crop_width)),
            ['Crop size greater than the image size.'])

        asserts = [rank_assertions[0], crop_size_assert]

        for i in range(1, len(image_list)):
            image = image_list[i]
            asserts.append(rank_assertions[i])
            with tf.control_dependencies([rank_assertions[i]]):
                shape = tf.shape(image)
            height = shape[0]
            width = shape[1]

            height_assert = tf.Assert(
                tf.equal(height, image_height),
                ['Wrong height for tensor %s [expected][actual]',
                 image.name, height, image_height])
            width_assert = tf.Assert(
                tf.equal(width, image_width),
                ['Wrong width for tensor %s [expected][actual]',
                 image.name, width, image_width])
            asserts.extend([height_assert, width_assert])

        # Create a random bounding box.
        #
        # Use tf.random_uniform and not numpy.random.rand as doing the former would
        # generate random numbers at graph eval time, unlike the latter which
        # generates random numbers at graph definition time.
        with tf.control_dependencies(asserts):
            max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        with tf.control_dependencies(asserts):
            max_offset_width = tf.reshape(image_width - crop_width + 1, [])
        offset_height = tf.random_uniform(
            [], maxval=max_offset_height, dtype=tf.int32)
        offset_width = tf.random_uniform(
            [], maxval=max_offset_width, dtype=tf.int32)

        return [self._crop(image, offset_height, offset_width,
                      crop_height, crop_width) for image in image_list]

    def image_processing(self, filename):

        if not self.custom_dataset :
            x_decode = filename
        else :
            x = tf.read_file(filename)
            x_decode = tf.image.decode_jpeg(x, channels=self.channels)

        #img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        shape = tf.shape(x_decode)
        img = tf.cond(tf.logical_and(tf.greater_equal(shape[0], self.load_size), tf.greater_equal(shape[1], self.load_size)),
                      lambda : self._random_crop([x_decode], crop_height=self.load_size, crop_width=self.load_size)[0],
                      lambda : tf.image.resize_images(x_decode, [self.load_size, self.load_size]))
        img = img.set_shape(self.load_size, self.load_size, self.channels)
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img


def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    x = np.concatenate((train_data, test_data), axis=0)
    x = np.expand_dims(x, axis=-1)

    return x

def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    x = np.concatenate((train_data, test_data), axis=0)

    return x

def load_data(dataset_name) :
    if dataset_name == 'mnist' :
        x = load_mnist()
    elif dataset_name == 'cifar10' :
        x = load_cifar10()
    else :

        #x = glob(os.path.join("./dataset", dataset_name, '*.*'))
        x = []
        for dirs, _, files in os.walk(os.path.join("./dataset", dataset_name)):
            for file in files:
                x.append(os.path.join(dirs, file))

    return x


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

##################################################################################
# Regularization
##################################################################################

def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully