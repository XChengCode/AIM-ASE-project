import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, tolerance=0):
        self.patience = patience
        self.tolerance = tolerance
        self.epoch_counter = 0
        self.max_validation_acc = np.NINF
    
    def should_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.epoch_counter = 0
        elif validation_acc < (self.max_validation_acc - self.tolerance):
            self.epoch_counter +=1
            if self.epoch_counter >= self.patience:              
                return True
        return False