import tensorflow as tf

from spektral.layers.convolutional import gcn_conv


class GCN(tf.keras.Model):

    def __init__(
        self,
        fcnnodes = 64,
        channels=16,
        activation="relu",
        output_activation="softmax",
        use_bias=False,
        dropout_rate=0.5,
        l1_reg=2.5e-4,
        n_input_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fcnnodes = fcnnodes
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.n_input_channels = n_input_channels
        reg = tf.keras.regularizers.l1(l1_reg)
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn1 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn2 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn3 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn4 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn5 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn6 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._gcn7 = gcn_conv.GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._den0 = tf.keras.layers.Dense(self.fcnnodes,activation = 'relu', kernel_regularizer = reg)
        self._den1 = tf.keras.layers.Dense(1)

    def get_config(self):
        return dict(
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            n_input_channels=self.n_input_channels,
        )

    def call(self, inputs):
        x1, a1 = inputs
        if self.n_input_channels is None:
            self.n_input_channels = x1.shape[-1]
        else:
            assert self.n_input_channels == x1.shape[-1]
        x1 = self._d0(x1)
        x1 = self._gcn0([x1, a1])
        gcnout = self._d1(x1)

        out = tf.keras.layers.Flatten()(gcnout)
        o1 = self._den0(out)
        o2 = self._den1(o1)
        return o2
