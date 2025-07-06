data = readmatrix("\data\output\midi_csv.csv");
fs = 44100;  % sampling rate
sound_cells = {};  % store each wave segment
pause = zeros(1, round(0.01 * fs));  % short pause between notes

for i = 1:size(data,1)
    pitch = data(i,1);
    start_time = data(i,2);
    end_time = data(i,3);
    velocity = data(i,4);

    freq = 440 * 2^((pitch - 69) / 12);
    t = 0:1/fs:(end_time - start_time);
    wave = (velocity / 127) * sin(2 * pi * freq * t);

    % append note and pause to the cell array
    sound_cells{end+1} = wave;
    sound_cells{end+1} = pause;
end

% Concatenate all wave segments into one signal
sound_wave = [sound_cells{:}];

% Play and save
%soundsc(sound_wave, fs);
audiowrite('data\output\generated.wav', sound_wave, fs);  % Save for spectrogram

[audio, fs] = audioread('data\output\generated.wav');
spectrogram(audio, 512, 256, 512, fs, 'yaxis');
title('Generated MIDI Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (kHz)');
colorbar;
saveas(gcf, 'data\output\spectrogram.png');