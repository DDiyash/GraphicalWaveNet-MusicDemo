# Graphical WaveNet MIDI Generator

This project uses a Graphical WaveNet model implemented in TensorFlow to generate classical music from MIDI files. It includes preprocessing (MIDI → Tensors), training, and autoregressive generation with MIDI output and MATLAB audio visualization.

Why I Didn't Use MATLAB for Model Development
I didn't use MATLAB for building or training the model because of the following limitations:

1.Lack of Support for Graph-Based Models:
MATLAB's Deep Learning Toolbox does not support advanced architectures like Graph Neural Networks (GNNs) or Graph WaveNet. These models rely on operations such as graph convolutions and temporal dilations, which are not natively available in MATLAB.

2.No Direct Way to Import Custom Models:
Graph WaveNet, implemented in TensorFlow with custom layers (e.g., gated temporal convolutions, einsum-based adjacency operations), cannot be easily converted to a MATLAB-compatible format like ONNX due to its non-standard components.

3.Limited Autoregressive Capabilities:
MATLAB does not support dynamic autoregressive generation and probabilistic sampling in the way required for symbolic music generation.

Why I Used MATLAB Only for Final Audio Synthesis and Visualization
Despite its modeling limitations, I used MATLAB only for the following tasks:

1.MIDI-to-WAV Conversion:
After generating a .mid file using Python, I converted it into audio by reading the MIDI data into a CSV format and synthesizing waveform segments based on pitch, velocity, and duration.

2.Spectrogram Visualization:
MATLAB provided a simple and effective way to generate a high-quality spectrogram of the audio, useful for evaluating the structure and content of the generated music.

## Requirements
```bash
pip install -r requirements.txt

## Run
python run_pipeline.py --midi_folder /data/surname_checked_midis --output_dir /data/output --output_midi generated.mid --steps 100 --epochs 20


## What to Do Next
Once you've generated the MIDI output, follow these steps to convert and visualize it:

1. Convert .mid to .csv for MATLAB
```bash
python matlab/midi_to_csv.py --input data/output/generated.mid --output data/output/midi_csv.csv

This creates a CSV file that contains the pitch, start time, end time, and velocity of each note.

2. Open MATLAB and Run the Conversion Script
Make sure your CSV file is in the correct path (./matlab/midi_csv.csv). Then:
Open csv_to_wav.m in MATLAB
Run it (F5 or Run button)

This will: Synthesize the .wav audio from the MIDI data, generate a spectrogram visualization, save the outputs as: generated.wav and spectrogram.png

Check the output folder in data folder for all the results.