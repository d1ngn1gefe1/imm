import os
import numpy as np
import tensorflow as tf
from imm.datasets.tps_dataset import TPSDataset
from datasets.vis import Vis


def load_dataset(data_root, subset):
  assert subset == 'train' or subset == 'valid', 'dataset split {} unavailable'.format(subset)

  image_dir = os.path.join(data_root, subset, 'JPEGImages')
  keypoints = None
  vis = Vis(data_root, subset)
  images = list(vis)

  return image_dir, images, keypoints


class VisDataset(TPSDataset):
  def __init__(self, data_dir, subset, max_samples=None,
               image_size=[128, 128], order_stream=False, landmarks=False,
               tps=True, vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='VisDataset'):

    super(VisDataset, self).__init__(
        data_dir, subset, max_samples=max_samples,
        image_size=image_size, order_stream=order_stream, landmarks=landmarks,
        tps=tps, vertical_points=vertical_points,
        horizontal_points=horizontal_points, rotsd=rotsd, scalesd=scalesd,
        transsd=transsd, warpsd=warpsd, name=name)

    self._image_dir, self._images, self._keypoints = load_dataset(data_dir, subset)


  def _get_sample_dtype(self):
    d =  {'image': tf.float32}
    return d


  def _get_sample_shape(self):
    d = {'image': [128, 128, 3]}
    return d


  def _proc_im_pair(self, inputs):
    with tf.name_scope('proc_im_pair'):
      height, width = self._image_size[:2]

      # read in the images:
      image = inputs['image']

      assert self._image_size[0] == self._image_size[1]
      final_size = self._image_size[0]

      image = tf.image.resize_images(
          image, [final_size, final_size], tf.image.ResizeMethod.BILINEAR,
          align_corners=True)

      mask = self._get_smooth_mask(height, width, 10, 20)[:, :, None]

      future_image = image

      inputs = {k: inputs[k] for k in self._get_sample_dtype().keys()}
      inputs.update({'image': image, 'future_image': future_image,
                     'mask': mask})
    return inputs

  def _get_image(self, idx):
    image = self._images[idx]
    inputs = {'image': image}
    return inputs
