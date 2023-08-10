import tensorflow as tf
import tensorflow_io as tfio


LABELS = ['ambulance', 'road']

def split_dataset(dataset, train_proportion = 0.65, test_proportion = 0.25, 
                    val_proportion = .10, shuffle=True, balance = False, buffer_size = 1000, seed = 42):
    '''Splits a dataset into train test and validation'''
    length = len(dataset)
    train_size = int(train_proportion * length)
    val_size = int(val_proportion* length)
    test_size = int(test_proportion * length)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed)
        
    train = dataset.take(train_size)
    validation = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size).skip(val_size)

    return train, validation, test


def compute_linear_matrix(downsampling_rate, num_mel_bins, lower_frequency, upper_frequency, frame_length):
    '''Computes the matrix for the linear to mel transformation'''
    num_spectrogram_bins = frame_length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=downsampling_rate,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    return linear_to_mel_weight_matrix

def get_label(filename):
    '''Gets the label from filename'''
    label = tf.strings.split(tf.strings.split(filename, '/')[1], '.')[0] 
    if tf.strings.length(label) < len("ambulance"):
        label = tf.strings.substr(label, 0, len("road"))
    else:
        label = tf.strings.substr(label, 0, len("ambulance"))
    
    return label

def get_mfccs_training(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients):
    '''From the filename gets the mfcc spectrogram and the corresponding label'''
    #read the audio and get the corresponding label
    audio_binary = tf.io.read_file(filename)
    label = get_label(filename)

    #gets the spectrogram from the binary audio
    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    #compute linear_to_mel_weigth_matrix
    weight_matrix = compute_linear_matrix(downsampling_rate, num_mel_bins, lower_frequency, upper_frequency, frame_length)

    #get spectrogram
    mfccs = get_mfcc(audio_binary, downsampling_rate, frame_length, frame_step, weight_matrix)

    return mfccs, label


def get_mfcc(audio_binary, downsampling_rate, frame_length, frame_step, weight_matrix):
    '''Common preprocessing used in training and testing (expecially por preprocessing latency evaluation)'''
    #get audio tensor
    audio, sampling_rate = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio)

    #peform padding
    zero_padding = tf.zeros(3*sampling_rate - tf.shape(audio), dtype=tf.float32)
    audio_padded = tf.concat([audio, zero_padding], axis=0)

    #get mfccs
    stft = tf.signal.stft(
        audio_padded,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.matmul(spectrogram, weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    return mfccs
