import os
import uuid
import json
import numpy as np
import tensorflow as tf 
import sounddevice as sd
import tensorflow_io as tfio
import paho.mqtt.client as mqtt 

from zipfile import ZipFile
from time import time

def get_audio_from_numpy(indata):
    '''transforms a numpy array into an audio tensor'''
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2*((indata + 32768) / (32767 + 32768)) -1
    indata = tf.squeeze(indata) 
    return indata


def get_spectrogram(audio, frame_length, frame_step):
    '''computes the spectrogram of an audio tensor'''
    zero_padding = tf.zeros(3*48000 - tf.shape(audio), dtype=tf.float32)
    audio_padded = tf.concat([audio, zero_padding], axis=0)
    stft = tf.signal.stft(
        audio_padded,
        frame_length=frame_length, #length of convolutional window
        frame_step=frame_step, #length of the step the convolutional window makes
        fft_length=frame_length #equal to frame length
    )
    spectrogram = tf.abs(stft)   
    return spectrogram


def is_silence(indata, frame_length, frame_step, dbFSthresh, duration_time):
    '''checks whether the audio is silent or not, hence whether its energy is below a certain threshold'''
    audio = get_audio_from_numpy(indata) 
    spectrogram = get_spectrogram(audio, frame_length, frame_step)

    dbFS = 20* tf.math.log(spectrogram + 1.e-6) 
    energy = tf.math.reduce_mean(dbFS, axis = 1) 
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1)* frame_length

    if non_silence_duration > duration_time:
        return False, spectrogram
    else:
        return True, spectrogram


def  get_mfccs_from_spectrogram(spectrogram, linear_to_mel_weight_matrix):

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    mfccs = tf.expand_dims(mfccs, 0)  # batch axis
    mfccs = tf.expand_dims(mfccs, -1)  # channel axis
    mfccs = tf.image.resize(mfccs, [32,32])

    return mfccs

def publisher(is_ambulance, MAC_ADDRESS):
    global client
    output_dict = {
        "mac_address": MAC_ADDRESS,
        "timestamp" : int(time()*1000),
        "ambulance" : is_ambulance, 
        }
    output_string = json.dumps(output_dict)
    client.publish('s291871', output_string)


#******************************************************************************************************************************************
#******************************************************************************************************************************************
MODEL_NAME = '1679338683784' 
ZIP_PATH = os.path.join(os.getcwd(), f'{MODEL_NAME}.tflite.zip')
MODEL_PATH = os.path.join(os.getcwd(), f'{MODEL_NAME}.tflite') 
blocksize = 48000 * 3
channels = 1
device = 1
samplerate = 48000
resolution = "int16"

PREPROCESSING_ARGS = {
'downsampling_rate': 48000,
'frame_length_in_s': .016,
'frame_step_in_s': .016,
'num_mel_bins': 20,
'lower_frequency': 20,
'upper_frequency': 8000
}
durationTime = 0.04
dbFsthresh = -120


#GET WEIGHT MATRIX
sampling_rate_float32 = tf.cast(PREPROCESSING_ARGS['downsampling_rate'], tf.float32)
frame_length = int(sampling_rate_float32 * PREPROCESSING_ARGS['frame_length_in_s'])
frame_step = int(sampling_rate_float32 * PREPROCESSING_ARGS['frame_step_in_s'])
num_spectrogram_bins = frame_length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=PREPROCESSING_ARGS['num_mel_bins'],
    num_spectrogram_bins=num_spectrogram_bins,
    sample_rate=PREPROCESSING_ARGS['downsampling_rate'],
    lower_edge_hertz=PREPROCESSING_ARGS['lower_frequency'],
    upper_edge_hertz=PREPROCESSING_ARGS['upper_frequency']
)


#MODEL 
with ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(os.getcwd())

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Allocated intepreter**********************************************")

#CONNECT TO HOST
client = mqtt.Client() 

#Pahoo mandatory method
def on_connect(client, userdata, flags, rc):
    print(f'Connected with result code {str(rc)}')

#Bind the on_connect method to the client
client.on_connect = on_connect
client.connect('mqtt.eclipseprojects.io', 1883)

#CREATE SERIES
MAC_ADDRESS = hex(uuid.getnode())

print("START LISTENING")


def callback(indata, frames, callback_time, status):
    '''call back function for sd.InputStream which stores the audio if not silent'''
    global PREPROCESSING_ARGS, frame_length, frame_step, interpreter, input_details, output_details, linear_to_mel_weight_matrix, dbFsthresh, durationTime
    silent, spectrogram = is_silence(indata, frame_length, frame_step, dbFsthresh, durationTime)

    if not silent:
        mfccs = get_mfccs_from_spectrogram(spectrogram, linear_to_mel_weight_matrix)

        #test the audio
        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        max_probability = max(output[0])
        top_index = np.argmax(output[0])

        if max_probability> 0.90 and top_index == 0: #Ambulance detected
            print(f"Emergency Siren Vehicle Detected")
            publisher(1, MAC_ADDRESS)

        if max_probability> 0.90 and top_index == 1: #RoadNoise detected
            print(f"RoadNoise Detected")
            publisher(0, MAC_ADDRESS)


with sd.InputStream(device = device, channels = channels , samplerate= samplerate, dtype = resolution, callback=callback, blocksize= blocksize):
    # Keeps recording until a ctrl+c keyboard interrupt occurs
    while True:
        continue


         
         