import tensorflow as tf
import tensorflow.keras as keras


class NSTModel(keras.Model):
    def __init__(self):
        super(NSTModel, self).__init__()
        with tf.compat.v1.variable_scope(name_or_scope="NST_VGG"):
            self.vgg19 = tf.keras.applications.VGG19(include_top=False)
            self.vgg19.trainable = False
        self.needed_feuatures = ["block1_conv1",
                                 "block2_conv1",
                                 "block3_conv1",
                                 "block4_conv1",
                                 "block5_conv1"]

    def call(self , inputs):
        extracted_features = []
        for layer in self.vgg19.layers:
            inputs = layer(inputs)

            if layer.name in self.needed_feuatures:
                extracted_features.append(inputs)

        return extracted_features

def loss_function(original_features, style_features, generated_features):

    original_loss = style_loss = 0

    for orig_feature, style_feature, gen_feature in zip(
        original_features , style_features , generated_features
    ):
        # batch_size will just be 1
        batch_size, height, width, channel = gen_feature.shape

        #original loss
        original_loss += tf.math.reduce_mean((gen_feature - orig_feature) ** 2)
        del orig_feature

        #Major error permute the given tensor
        gen_feature = tf.reshape(gen_feature, shape=(channel, height * width))
        style_feature = tf.reshape(style_feature, shape=(channel, height * width))

        G = tf.matmul(a=gen_feature,
                      b=gen_feature,
                      transpose_b=True)
        # Compute Gram Matrix of Style
        A = tf.matmul(a=style_feature,
                      b=style_feature,
                      transpose_b=True)
        style_loss += tf.math.reduce_mean((G - A) ** 2)
        del gen_feature, style_feature

    alpha = 0.01
    beta = 100

    total_loss = alpha * original_loss + beta * style_loss
    return total_loss