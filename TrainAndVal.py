import torch
import Checkpointing

def train_step(epoch, model, data_loader, loss_fn, optimizer, 
lr_scheduler, acc_fn, writer, DEVICE):
 
    train_loss = 0  
    train_acc = 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X  = X.to(DEVICE)
        y  = y.to(DEVICE)
        y_pred = model(X)
        
        loss = loss_fn(y_pred,y)
        train_loss += loss
        train_acc += acc_fn(y_pred, y) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(data_loader)
    train_acc  /= len(data_loader)
    
    
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}% | LR: {lr_scheduler.get_last_lr()[0]:.5f}")
    
    writer.add_scalar(f"Loss/train", train_loss,epoch)
    writer.add_scalar(f"Acc/train", train_acc,epoch)

    if(lr_scheduler.get_last_lr()[0]>=0.00001):
        lr_scheduler.step()

def val_step(epoch, model, data_loader, loss_fn, 
early_stopper, acc_fn, writer, DEVICE):
 
    val_loss = 0 
    val_acc = 0
  
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X  = X.to(DEVICE)
            y = y.to(DEVICE)
            val_pred = model(X)

            val_loss += loss_fn(val_pred, y)
            val_acc += acc_fn(val_pred, y)
      
        val_loss /= len(data_loader)
        val_acc /= len(data_loader)

       
        print(f"Validation loss: {val_loss:.5f} | Validation acc: {val_acc:.2f}\n")

        writer.add_scalar(f"Loss/val", val_loss,epoch)
        writer.add_scalar(f"Acc/val", val_acc,epoch)

        
        if early_stopper.should_stop(val_acc):
            print(f"\nValidation accuracy has not improved for {early_stopper.epoch_counter} epoch, aborting...")
            return 0
        else:
            if early_stopper.epoch_counter > 0:
                print (f"Epochs without improvement: {early_stopper.epoch_counter}\n")
            return 1