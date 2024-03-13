#importing libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product

#setting the parameters on matplotlib
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

#define the kernel
kernel = tf.constant([[-1,-1,-1], [-1,8,-1], [-1, -1, -1]])

#load the image
image = tf.io.read_file('beach.jpeg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size = [300, 300])

#plot the image
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Gray Scale image')
plt.show()

#reformat
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel,[*kernel.shape,1,1])
kernel = tf.cast(kernel, dtype=tf.float32)

#convolution layer
conv_fn = tf.nn.conv2d
image_filter = conv_fn(
    input = image,
    filters = kernel,
    strides = 1, # or (1,1)
    padding = 'SAME',
)
plt.figure(figsize=(15,5))

#plot the convolved image
plt.subplot(1, 3, 1)

plt.imshow(
    tf.squeeze(image_filter)
)
plt.axis('off')
plt.title('Convolution')
plt.show()