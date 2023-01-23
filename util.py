import torch
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import random_split
import torch.nn.functional as F
import librosa
device ='cuda' if torch.cuda.is_available () else 'cpu'


def get_padding(path):
    audio, sr = librosa.load(path)      
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=2048,hop_length=512,n_mels=64)   
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)   
    tensor_log_mel_spectrogram = torch.tensor(log_mel_spectrogram)        
    padding_log_mel_spectrogram= F.pad(tensor_log_mel_spectrogram, (0,240-tensor_log_mel_spectrogram.shape[1],0,0))
    return padding_log_mel_spectrogram

def get_split(dataset, train_test_split=[0.8, 0.2]):
    tr_te = []
    for frac in train_test_split:   # Calculate the number of data in train and test part
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_te.append(actual_count)
    train_split, test_split = random_split(dataset, tr_te)
    return train_split, test_split

def get_checkpoint(filepath):
    l=[]
    for i in filepath:
        l.append(i)
    return len(l)


def get_acc(model, dataset, criterion):
    test_loss = []
    test_accs = []
    for batch in tqdm(dataset):
        imgs, labels = batch
        with torch.no_grad ():
            logits = model (imgs.to (device))
            loss = criterion (logits, labels.to (device))
            acc = (logits.argmax (dim=-1) == labels.to (device)).float ().mean ()
            test_loss.append(loss.item ())
            test_accs.append(acc)
    test_acc = sum (test_accs) / len (test_accs)
    return test_acc