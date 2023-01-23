import torch
import torch.nn.functional as F
import torch.nn as nn
import data
import Load_data

import torchvision

import unittest
from PIL import Image
from torch.utils.data import DataLoader

from util import get_acc, get_padding, get_split, get_checkpoint
from ResNet import *


class TestProject(unittest.TestCase):
    
    def test_padding(self):
        test_path = r'RAVDESS_wave_data/Actor_12/03-01-06-01-02-01-12.wav'
        spectrogram = get_padding(test_path)
        self.assertEqual(spectrogram.shape[0], 64)
        self.assertEqual(spectrogram.shape[1], 240)
        
    def test_split(self):
        dataframe = data.EMODB_collate("EMO-DB_wave_data/wav")
        dataset = Load_data.AudioDataset(dataframe)
        train_split, test_split = get_split(dataset, train_test_split=[0.8, 0.2])
        self.assertEqual(len(train_split)+len(test_split), len(dataset))
        
    def test_checkpoint(self):
        filepath = "saved_models"
        l = get_checkpoint(filepath)
        self.assertGreater(l, 0)
                
    def test_acc(self):
        test_path = r'CASIA_wave_data/liuchanhg'
        model_path = r'saved_models/ResNet_checkpoint_45.pt'

        df = data.RAVDESS_collate("RAVDESS_wave_data/")
        _, test_dl = Load_data.load_data(dataframe=df, batch_sz=32)
        loss_fn = nn.CrossEntropyLoss ()
        model=torch.load(f=model_path)
        acc = get_acc(model, test_dl, loss_fn) 
        self.assertGreater(acc, 0.5)

        
        
if __name__ == '__main__':
    unittest.main()
            
        
        