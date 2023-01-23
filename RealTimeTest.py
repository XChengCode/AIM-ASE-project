import torch.nn.functional as F
import multiprocessing as mp
import torch
import sounddevice as sd
import librosa.display
import librosa

def RealTimeAudioTest(model):


    duration = 4  # seconds
    fs = 22050
    # Start recording
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    print("Recording started.")
    sd.wait()  # Wait for the recording to finish
    print("Recording finished.")

    mel_spectrogram = librosa.feature.melspectrogram(y=recording.T,n_fft=2048,hop_length=512,n_mels=64)   
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)   
    tensor_log_mel_spectrogram = torch.tensor(log_mel_spectrogram)   
    tensor_log_mel_spectrogram  = (tensor_log_mel_spectrogram - torch.mean(tensor_log_mel_spectrogram)) / torch.std(tensor_log_mel_spectrogram)
    padding_log_mel_spectrogram= F.pad(tensor_log_mel_spectrogram, (0,240-tensor_log_mel_spectrogram.shape[1],0,0),"constant",0)
 
    padding_log_mel_spectrogram = padding_log_mel_spectrogram

    padding_log_mel_spectrogram = padding_log_mel_spectrogram.unsqueeze(0)

    val_pred = model(padding_log_mel_spectrogram)
    model_pred_probs = torch.softmax(val_pred, dim=1)
    model_pred_label = torch.argmax(model_pred_probs, dim=1)

    labels=['neutral','calm','happy','sad','angry','fear','disgust','surprise']

    return labels[model_pred_label]


def AudioTest(path, model):


    audio,_ = librosa.load(path)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio,n_fft=2048,hop_length=512,n_mels=64)   
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)   
    tensor_log_mel_spectrogram = torch.tensor(log_mel_spectrogram)   
    tensor_log_mel_spectrogram  = (tensor_log_mel_spectrogram - torch.mean(tensor_log_mel_spectrogram)) / torch.std(tensor_log_mel_spectrogram)
    padding_log_mel_spectrogram= F.pad(tensor_log_mel_spectrogram, (0,240-tensor_log_mel_spectrogram.shape[1],0,0),"constant",0)

    padding_log_mel_spectrogram = padding_log_mel_spectrogram.unsqueeze(0).unsqueeze(0)

    val_pred = model(padding_log_mel_spectrogram)
    model_pred_probs = torch.softmax(val_pred, dim=1)
    model_pred_label = torch.argmax(model_pred_probs, dim=1)

    labels=['neutral','calm','happy','sad','angry','fear','disgust','surprise']

    return labels[model_pred_label]
    

       