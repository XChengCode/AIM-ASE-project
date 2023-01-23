import os
import torch

def TinyVGG_save_checkpoint(model, epoch, save_dir):
    filename = f"TinyVGG_checkpoint_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)


def ResNet_save_checkpoint(model, epoch, save_dir):
    filename = f"ResNet_checkpoint_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)
    
    
def Final_ResNet_save_checkpoint(model, epoch, save_dir):
    filename = f"Final_ResNet_checkpoint_{epoch}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(model, save_path)