import numpy as np
import tensorflow as tf
from model.model import GraphicalWaveNet

def train_model(X, A, epochs=10, batch_size=1, lr=1e-3, save_path='gwnet_model.weights.h5'):
    B, T, N, F = X.shape
    model = GraphicalWaveNet(F, F, A)
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(y, y_pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        losses = []
        for i in range(0, B, batch_size):
            xb = X[i:i+batch_size, :-1]
            yb = X[i:i+batch_size, 1:]
            loss = train_step(xb, yb)
            losses.append(loss.numpy())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.6f}")

    model.save_weights(save_path)
    return model

def load_single_x(path):
    x = np.load(path).astype(np.float32)   # (N, T, F)
    x = np.transpose(x, (1, 0, 2))         # (T, N, F)
    return np.expand_dims(x, axis=0)       # (1, T, N, F)

def load_adjacency(path):
    return np.load(path).astype(np.float32)
