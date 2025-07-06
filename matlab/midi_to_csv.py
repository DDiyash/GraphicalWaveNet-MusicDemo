import pretty_midi
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

midi = pretty_midi.PrettyMIDI(args.input)
print(midi)

notes = []
for inst in midi.instruments:
    for note in inst.notes:
        notes.append([note.pitch, note.start, note.end, note.velocity])

print(notes[:5])
data = pd.DataFrame(notes, columns=['pitch', 'start', 'end', 'velocity'])
print(data.head())

data.to_csv(args.output, index=False)
print(f"Saved MIDI as CSV to {args.output}")
