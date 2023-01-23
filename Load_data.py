from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import multiprocessing as mp
import torch
import librosa

class AudioDataset(Dataset):
    def __init__(self, dataframe): 
        super().__init__()
        self.data = self.prepare_wavs(dataframe)
        
    def prepare_wavs(self, df):
        l = []
        for idx, i in enumerate(df["Audio Path"]):   # Traversing the entire database
            # Sampling the audio data
            audio, sr = librosa.load(i)   
            # Convert it into Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=2048,hop_length=512,n_mels=64)   
            # Execute logarithmic operations
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)   
            # Convert numpy array into tensor
            tensor_log_mel_spectrogram = torch.tensor(log_mel_spectrogram)   
            # Data normalization
            tensor_log_mel_spectrogram  = (tensor_log_mel_spectrogram - torch.mean(tensor_log_mel_spectrogram)) / torch.std(tensor_log_mel_spectrogram)
            # Spectrogram padding
            padding_log_mel_spectrogram= F.pad(tensor_log_mel_spectrogram, (0,240-tensor_log_mel_spectrogram.shape[1],0,0),"constant",0)
            # Combine the tensors with labels and put it into the list
            label = df["Labels"][idx]
            l.append((padding_log_mel_spectrogram,label))
            print(f"{idx+1}/{len(df)} ({100*(idx+1)/len(df):.1f}%)", end = "\r" if idx < len(df) else "\n", flush=True)
        return l

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        return self.data[i]
    
    
    
def custom_collate_fn(batch):
    image_tensors = []
    labels = []
    for image, label in batch:
        image_tensor = image.unsqueeze(0)  # Add 1 dimension at the beginning
        image_tensors.append(image_tensor)
        labels.append(label)

    image_batch_tensor = torch.stack(image_tensors)  # Convert list into tensor
    label_batch_tensor = torch.LongTensor(labels)
    return (image_batch_tensor,label_batch_tensor)

def load_data(dataframe, batch_sz=100, train_test_split=[0.8, 0.2]):
    dataset = AudioDataset(dataframe)  # Define a member of the subclass
    tr_te = []
    for frac in train_test_split:   # Calculate the number of data in train and test part
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_te.append(actual_count)
    train_split, test_split = random_split(dataset, tr_te)
    num_cpus = mp.cpu_count()
    
    # Split the dataset into batches
    train_dl = DataLoader(train_split, batch_size=batch_sz,collate_fn=custom_collate_fn,shuffle=True,num_workers=num_cpus)
    test_dl  = DataLoader(test_split,  batch_size=batch_sz,collate_fn=custom_collate_fn,shuffle=True,num_workers=num_cpus)
    return train_dl, test_dl