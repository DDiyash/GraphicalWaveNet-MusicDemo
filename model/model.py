import tensorflow as tf

class GatedTemporalConv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=2, dilation_rate=1):
        super().__init__()
        self.conv_tanh = tf.keras.layers.Conv1D(
            channels, kernel_size, dilation_rate=dilation_rate,
            padding='causal', activation='tanh')
        self.conv_sigmoid = tf.keras.layers.Conv1D(
            channels, kernel_size, dilation_rate=dilation_rate,
            padding='causal', activation='sigmoid')

    def call(self, x_a, x_b):
        return self.conv_tanh(x_a) * self.conv_sigmoid(x_b)

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.linear = tf.keras.layers.Dense(channels)

    def call(self, x, A):
        return self.linear(tf.einsum('ij,btjf->btif', A, x))

class GraphicalWaveNet(tf.keras.Model):
    def __init__(self, in_channels, out_channels, adj_matrix, dilation_depth=4, hidden_channels=32):
        super().__init__()
        self.A = tf.constant(adj_matrix, dtype=tf.float32)
        self.in_proj = tf.keras.layers.Dense(hidden_channels * 3)

        self.tcn_blocks = []
        self.gcn_blocks = []
        self.residual_projs = []
        self.skip_conns = []

        for i in range(dilation_depth):
            self.tcn_blocks.append(GatedTemporalConv(hidden_channels, dilation_rate=2**i))
            self.gcn_blocks.append(GraphConvolution(hidden_channels))
            self.residual_projs.append(tf.keras.layers.Dense(hidden_channels))
            self.skip_conns.append(tf.keras.layers.Dense(hidden_channels))

        self.output_stack = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_channels),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(out_channels)
        ])

    def call(self, X):
        B, T, N, F = X.shape
        X_proj = self.in_proj(X)
        x_a, x_b, x_res = tf.split(X_proj, 3, axis=-1)

        skip_outputs = []
        for tcn, gcn, res_proj, skip_proj in zip(
            self.tcn_blocks, self.gcn_blocks, self.residual_projs, self.skip_conns):

            x_a_r = tf.reshape(x_a, [B * N, T, -1])
            x_b_r = tf.reshape(x_b, [B * N, T, -1])
            tcn_out = tcn(x_a_r, x_b_r)
            tcn_out = tf.reshape(tcn_out, [B, N, T, -1])
            tcn_out = tf.transpose(tcn_out, [0, 2, 1, 3])

            gcn_out = gcn(x_res, self.A)
            h = tcn_out + gcn_out

            x_res = res_proj(h) + x_res
            skip_outputs.append(skip_proj(h))

        total_skip = tf.reduce_sum(tf.stack(skip_outputs), axis=0)
        return self.output_stack(total_skip)
