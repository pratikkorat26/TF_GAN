import tensorflow as tf

'''
Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1

Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1
'''
def DownConvolution(inputs,
                    filters,
                    kernel_size,
                    strides,
                    activation,
                    name):
    with tf.compat.v1.variable_scope(name_or_scope=name):

        net = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     name=f"{name}/conv2d")(inputs)
        net = tf.keras.layers.BatchNormalization(axis=-1,
                                                 momentum=0.999,
                                                 epsilon=0.001,
                                                 name=f"{name}/batch_norm")(net)
        net = tf.keras.layers.Activation(activation=activation,
                                         name=f"{name}/{activation}")(net)

    return net

def DownConvolutionWithLeakyRelu(inputs,
                    filters,
                    kernel_size,
                    strides,
                    name):
    with tf.compat.v1.variable_scope(name_or_scope=name):

        net = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     name=f"{name}/conv2d")(inputs)
        net = tf.keras.layers.BatchNormalization(axis=-1,
                                                 momentum=0.999,
                                                 epsilon=0.001,
                                                 name=f"{name}/batch_norm")(net)
        net = tf.keras.layers.LeakyReLU(name=f"{name}/LeakyReLU")(net)

    return net


def UpConvolution(inputs,
                    filters,
                    kernel_size,
                    strides,
                    activation,
                    name):
    with tf.compat.v1.variable_scope(name_or_scope=name, reuse=True):
        net = tf.keras.layers.Conv2DTranspose(filters=filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              name=f"{name}/conv_transpose")(inputs)
        net = tf.keras.layers.BatchNormalization(axis=-1,
                                                 momentum=0.999,
                                                 epsilon=0.001,
                                                 name=f"{name}/batch_norm")(net)
        net = tf.keras.layers.Activation(activation=activation,
                                         name=f"{name}/{activation}")(net)

    return net

def build_generator(latent_size=100):
    #Input dim specifying
    #for generator remeber always use RELU and TANh at the end
    with tf.compat.v1.variable_scope(name_or_scope = "gen" , reuse = None):
        inputs = tf.keras.layers.Input(shape=(latent_size,) , name = "input")
    #Initialier part of generator
        net = tf.keras.layers.Dense(units=4 * 4 * 128,
                                    kernel_regularizer=tf.keras.regularizers.l2(),
                                    name="dense_0")(inputs)
        net = tf.keras.layers.BatchNormalization(axis=-1,
                                                 momentum=0.999,
                                                 epsilon=0.001,
                                                 name="batch_norm_0")(net)
        net = tf.keras.layers.Activation(activation="relu",
                                         name="relu_0")(net)

        net = tf.keras.layers.Dense(units=7 * 7 * 64,
                                    kernel_regularizer=tf.keras.regularizers.l2(),
                                    name="dense_1")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1,
                                                 momentum=0.999,
                                                 epsilon=0.001,
                                                 name="batch_norm_1")(net)
        net = tf.keras.layers.Activation(activation="relu",
                                         name="relu_1")(net)

        #Reshaping the converted input
        net = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(net)

        #Upsampling the feature maps
        net = UpConvolution(net,
                            filters=32,
                            kernel_size=(3, 3),
                            strides=2,
                            activation="relu",
                            name="upsample_1")
        net = UpConvolution(net,
                            filters=16,
                            kernel_size=(3, 3),
                            strides=2,
                            activation="relu",
                            name="upsample_2")

        net = DownConvolution(net,
                              filters=1,
                              kernel_size=(4,4),
                              strides=1,
                              activation="tanh",
                              name="downconv_1")

    model = tf.keras.Model(inputs=inputs , outputs=net)

    return model

def build_discriminator(input_size=(28,28,1)):
    #build_discriminator
    with tf.compat.v1.variable_scope(name_or_scope="disc"):
        inputs = tf.keras.layers.Input(shape=(input_size))

        net = DownConvolutionWithLeakyRelu(inputs,
                                           filters=32,
                                           kernel_size=(3,3),
                                           strides=1,
                                           name="downsample_0")
        net = DownConvolutionWithLeakyRelu(net,
                                           filters=64,
                                           kernel_size=(4, 4),
                                           strides=2,
                                           name="downsample_1")
        net = DownConvolutionWithLeakyRelu(net,
                                           filters=128,
                                           kernel_size=(4, 4),
                                           strides=2,
                                           name="downsample_2")

        net = tf.keras.layers.Flatten()(net)

        net = tf.keras.layers.Dense(128)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.LeakyReLU(alpha = 0.1)(net)

        net = tf.keras.layers.Dense(1)(net)




    model = tf.keras.Model(inputs=inputs , outputs=net)

    return model




if __name__ == '__main__':
    pass

