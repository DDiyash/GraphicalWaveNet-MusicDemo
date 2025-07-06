import numpy as np
import tensorflow as tf
import pretty_midi

def generate_sequence(model, seed, A, steps=100):
    generated = []
    x = seed.copy()
    for _ in range(steps):
        y_pred = model(x)
        next_step = y_pred[:, -1:, :, :]

        on_prob = tf.sigmoid(next_step[..., 0])
        sampled_on = tf.cast(tf.random.uniform(on_prob.shape) < on_prob, tf.float32)
        sampled_vel = tf.clip_by_value(next_step[..., 1] + tf.random.normal(shape=next_step[..., 1].shape, stddev=0.05), 0, 1)

        sampled = tf.concat([sampled_on[..., None], sampled_vel[..., None]], axis=-1)
        if next_step.shape[-1] > 2:
            sampled = tf.concat([sampled, next_step[..., 2:]], axis=-1)

        x = tf.concat([x[:, 1:], sampled], axis=1)
        generated.append(sampled.numpy()[0])
    return np.concatenate(generated, axis=0)

def save_prediction_as_midi(pred, path, time_resolution=0.25):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    T, N, F = pred.shape
    for n in range(N):
        for t in range(T):
            on = pred[t, n, 0] > 0.5
            vel_raw = pred[t, n, 1]
            if np.isnan(vel_raw):
                vel_raw = 0.0
            vel = int(np.clip(vel_raw * 127, 0, 127))
            if on:
                note = pretty_midi.Note(
                    velocity=vel,
                    pitch=n,
                    start=t * time_resolution,
                    end=(t + 1) * time_resolution
                )
                inst.notes.append(note)
    midi.instruments.append(inst)
    midi.write(path)
