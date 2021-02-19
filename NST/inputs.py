import tensorflow as tf
import matplotlib.pyplot as plt

@tf.function
def get_input_images():
    def _parse_function(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded/255, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image
    with tf.compat.v1.variable_scope(name_or_scope = "original_image"):
        image = _parse_function("style.jpg")
        image = tf.image.resize(image,size = (400,400))
        image = tf.image.rot90(image,3)
    with tf.compat.v1.variable_scope(name_or_scope="style_image"):
        style_image = _parse_function("style.jpg")
        style = tf.image.resize(style_image, size=(400,400))
    return image, style


if __name__ == "__main__":
    #image,style_image = get_input_images()
    # image = tf.keras.applications.vgg19.preprocess_input(image)
    # style = tf.keras.applications.vgg19.preprocess_input(style_image)
    model = tf.keras.applications.VGG19(include_top = False)
    for layer in model.layers:
        print(layer.name)