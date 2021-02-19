import tensorflow as tf
from inputs import get_input_images
from nst import NSTModel , loss_function
import PIL
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    #hyperparameter
    total_steps = 6000
    learning_rate = 0.001

    #setting up the model
    model = NSTModel()

    #setting up the images:
    image , style = get_input_images()
    generated = tf.Variable(image , trainable = True)

    opt = tf.optimizers.Adam(learning_rate=0.02,
                             beta_1=0.99,
                             epsilon=1e-1)


    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    @tf.function()
    def train_step(image , style , generated):
        with tf.GradientTape() as tape:
            original_features = model(image)
            style_features = model(style)
            generated_features = model(generated)

            loss = loss_function(original_features,
                                 style_features,
                                 generated_features)

        grad = tape.gradient(loss, generated)
        opt.apply_gradients([(grad, generated)])
        generated.assign(clip_0_1(generated))


    for i in range(100):
        train_step(image , style , generated)
        if i % 10 == 0:
            final = tensor_to_image(generated)
            plt.imshow(final)
            plt.show()