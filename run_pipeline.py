import argparse
import numpy as np
import os
from model.model import GraphicalWaveNet
from model.generate import generate_sequence, save_prediction_as_midi
from model.train import train_model
from utils.midi_to_tensor import to_tensor
from utils.adjacency import adjacency


def main(args):
    # Convert MIDI files to tensor and adjacency matrix
    print("Converting MIDI to tensors...")
    X = to_tensor(args.midi_folder, time_resolution=args.time_resolution, max_time=args.max_time)
    A = adjacency()

    # Save tensors (optional)
    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "A.npy"), A)

    # Prepare tensors
    X = np.transpose(X, (1, 0, 2))  # (T, N, F)
    X = np.expand_dims(X, axis=0)   # (1, T, N, F)

    # Train the model
    print("Training model...")
    model = train_model(X, A, epochs=args.epochs, batch_size=1)

    # Autoregressive generation
    print("Generating new music...")
    T_seed = 16
    max_start = X.shape[1] - T_seed - 1
    start = np.random.randint(0, max_start)
    seed = X[:, start:start+T_seed]

    generated = generate_sequence(model, seed, A, steps=args.steps)
    midi_out_path = os.path.join(args.output_dir, args.output_midi)
    save_prediction_as_midi(generated, midi_out_path, time_resolution=args.time_resolution)
    print(f"Saved generated MIDI to {midi_out_path.replace(os.sep, '/')}")
    #print(f"Saved generated MIDI to {midi_out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_folder', type=str, required=True, help='Folder containing MIDI files')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save model and output')
    parser.add_argument('--output_midi', type=str, default='generated_autoregressive.mid', help='Name of output MIDI file')
    parser.add_argument('--time_resolution', type=float, default=0.2, help='Time step size in seconds')
    parser.add_argument('--max_time', type=int, default=100, help='Max time (in seconds) to represent in tensor')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to generate')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
