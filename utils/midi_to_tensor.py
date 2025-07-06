import os
import time
import numpy as np
import pretty_midi

def to_tensor(midi_folder, time_resolution=0.2, max_time=100):
    N = 128  # MIDI pitch range
    T = int(max_time / time_resolution)
    F = 4
    instruments = {}  # Map instrument program to index
    X = np.zeros((N, T, F))

    start = time.time()
    for i, filename in enumerate(os.listdir(midi_folder)):
        if not filename.endswith(('.mid', '.midi')):
            continue
        if i % 100 == 0:
            print(f"Processing {i}: {filename} (elapsed {int(time.time() - start)}s)")

        filepath = os.path.join(midi_folder, filename)
        try:
            pm = pretty_midi.PrettyMIDI(filepath)
            for inst in pm.instruments:
                instr_idx = instruments.setdefault(inst.program, len(instruments)) / 128.0
                for note in inst.notes:
                    pitch = note.pitch
                    start_step = int(note.start / time_resolution)
                    if start_step >= T:
                        continue
                    X[pitch, start_step, 0] = 1
                    X[pitch, start_step, 1] = note.velocity / 127
                    X[pitch, start_step, 2] = note.end - note.start
                    X[pitch, start_step, 3] = instr_idx
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
    return X
