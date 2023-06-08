from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class StyleTransfer():
    def __init__(self):
        self.hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    def load_img(self, decoded_img):
        max_dim = 512
        img = tf.io.decode_base64(decoded_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = tf.reduce_max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def neuralstyle(self, content_image, style_image):
        img1 = self.load_img(content_image)
        img2 = self.load_img(style_image)
        content_img = tf.constant(img1)
        style_img = tf.constant(img2)
        model = self.hub_model(content_img, style_img)[0]
        image = self.tensor_to_image(model)
        return image
