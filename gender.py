import tensorflow as tf
import numpy as np
import random
import os
import pathlib
import shutil
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SAMPLE_PATH = sys.argv[1]
new_model = tf.keras.models.load_model('test_model.h5')

# Получение спектограммы
def get_spectrogram(waveform, windows_size=255, hop_len=128):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
      equal_length, frame_length=windows_size, frame_step=hop_len)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


# Функция для декодирования wav файла
def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)





def inference(SAMPLE_PATH):
    with open(SAMPLE_PATH, 'br') as wav:
        wav_binary = wav.read()

    wav_decoded = decode_audio(wav_binary)

    spectrogram = get_spectrogram(wav_decoded, windows_size=255, hop_len=128)

    y_pred = np.argmax(new_model.predict(spectrogram[None, ...]), axis=1)
    if y_pred[0] == 1:
        predicted_gender = 'male'
    else:
        predicted_gender = 'female'
    return predicted_gender

print("Predicted gender:", inference(SAMPLE_PATH))