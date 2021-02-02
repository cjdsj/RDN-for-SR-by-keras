import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']= int(config['filters'] / self.r*self.r)
        config['r'] = self.r
        return config


class ResidualDenseBlock(Model):
    def __init__(self, num_G):
        super(ResidualDenseBlock, self).__init__()
        self.num_G = num_G

        self.c1 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')
        self.c2 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')
        self.c3 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')
        self.c4 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')
        self.c5 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')
        self.c6 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')

        self.c = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(1, 1), padding='same')

    def call(self, inputs):
        x1 = self.c1(inputs)
        y1 = tf.concat([inputs, x1], 3)

        x2 = self.c2(y1)
        y2 = tf.concat([inputs, x1, x2], 3)

        x3 = self.c3(y2)
        y3 = tf.concat([inputs, x1, x2, x3], 3)

        x4 = self.c4(y3)
        y4 = tf.concat([inputs, x1, x2, x3, x4], 3)

        x5 = self.c5(y4)
        y5 = tf.concat([inputs, x1, x2, x3, x4, x5], 3)

        x6 = self.c6(y5)
        y6 = tf.concat([inputs, x1, x2, x3, x4, x5, x6], 3)

        y = self.c(y6)
        return y + inputs


class RDN(Model):
    def __init__(self, num_G, channels, scale):
        super(RDN, self).__init__()
        self.num_G = num_G
        self.channels = channels
        self.scale = scale

        self.SFE1 = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')
        self.SFE2 = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')

        self.RDB1 = ResidualDenseBlock(self.num_G)
        self.RDB2 = ResidualDenseBlock(self.num_G)
        self.RDB3 = ResidualDenseBlock(self.num_G)
        self.RDB4 = ResidualDenseBlock(self.num_G)
        self.RDB5 = ResidualDenseBlock(self.num_G)
        self.RDB6 = ResidualDenseBlock(self.num_G)
        self.RDB7 = ResidualDenseBlock(self.num_G)
        self.RDB8 = ResidualDenseBlock(self.num_G)
        self.RDB9 = ResidualDenseBlock(self.num_G)
        self.RDB10 = ResidualDenseBlock(self.num_G)
        self.RDB11 = ResidualDenseBlock(self.num_G)
        self.RDB12 = ResidualDenseBlock(self.num_G)
        self.RDB13 = ResidualDenseBlock(self.num_G)
        self.RDB14 = ResidualDenseBlock(self.num_G)
        self.RDB15 = ResidualDenseBlock(self.num_G)
        self.RDB16 = ResidualDenseBlock(self.num_G)
        self.RDB17 = ResidualDenseBlock(self.num_G)
        self.RDB18 = ResidualDenseBlock(self.num_G)
        self.RDB19 = ResidualDenseBlock(self.num_G)
        self.RDB20 = ResidualDenseBlock(self.num_G)

        self.GFF1 = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(1, 1), padding='same')
        self.GFF2 = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')

        self.UP = Subpixel(64, (3,3), r=scale, padding='same',activation='relu')

        self.c = Conv2D(filters=self.channels, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')

    def call(self, inputs):
        sfe1 = self.SFE1(inputs)
        sfe2 = self.SFE2(sfe1)

        rdb1 = self.RDB1(sfe2)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        rdb4 = self.RDB4(rdb3)
        rdb5 = self.RDB5(rdb4)
        rdb6 = self.RDB6(rdb5)
        rdb7 = self.RDB7(rdb6)
        rdb8 = self.RDB8(rdb7)
        rdb9 = self.RDB9(rdb8)
        rdb10 = self.RDB10(rdb9)
        rdb11 = self.RDB11(rdb10)
        rdb12 = self.RDB12(rdb11)
        rdb13 = self.RDB13(rdb12)
        rdb14 = self.RDB14(rdb13)
        rdb15 = self.RDB15(rdb14)
        rdb16 = self.RDB16(rdb15)
        rdb17 = self.RDB17(rdb16)
        rdb18 = self.RDB18(rdb17)
        rdb19 = self.RDB19(rdb18)
        rdb20 = self.RDB20(rdb19)
        rdb = tf.concat([rdb1, rdb2, rdb3, rdb4, rdb5, rdb6, rdb7, rdb8, rdb9, rdb10,
                rdb11, rdb12, rdb13, rdb14, rdb15, rdb16, rdb17, rdb18, rdb19, rdb20], 3)

        gff1 = self.GFF1(rdb)
        gff2 = self.GFF2(gff1)
        dff = sfe1 + gff2

        up = self.UP(dff)

        y = self.c(up)
        return y


def L1_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))


if __name__=='__main__':
    model = RDN(num_G=32, channels=3, scale=2)
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=L1_loss)
    model.build((None, 32, 32, 3))
    model.summary()