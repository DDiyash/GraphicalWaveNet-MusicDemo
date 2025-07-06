import numpy as np
import pretty_midi

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
