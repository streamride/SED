import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter



def train(model, train_loader, loss_fn, optimizer, scheduler, epoch, tensorboard_writer):
    epoch_train_loss = 0
    train_metrics = 0.0
    model.train()
    for x, yl, yo in tqdm(train_loader, desc='training'):
        optimizer.zero_grad()
        
        _, predictions = model(x)
        yo = yo.squeeze()
        loss = loss_fn(predictions, yl)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        predictions = predictions.detach().numpy()
        predictions = np.argmax(predictions, axis=1)
        yl = yl.detach().numpy()
        
        f_score = metrics.f1_score(yl, predictions, average='macro')
    
        train_metrics += f_score
        tensorboard_writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], epoch)
        tensorboard_writer.add_scalar("Loss/train", loss, epoch)
        tensorboard_writer.add_scalar("F1Score/train", f_score, epoch)
        epoch_train_loss += loss.item()
    tqdm.write(f'Epoch {epoch}')
    tqdm.write(f'Train Epoch loss {epoch_train_loss / len(train_loader)}')
    tqdm.write(f'Train f1 score {train_metrics / len(train_loader)}')
    return train_metrics / len(train_loader), epoch_train_loss / len(train_loader)
    
def evaluate(model, test_loader, loss_fn, epoch, tensorboard_writer):
    model.eval()
    test_metrics = 0.0
    epoch_test_loss = 0.0
    for x, yl, yo in tqdm(test_loader, desc='eval'):
        _, predictions = model(x)
        yo = yo.squeeze()
        loss = loss_fn(predictions, yl)
        epoch_test_loss += loss.item()
        tensorboard_writer.add_scalar("Loss/eval", loss, epoch)
        predictions = predictions.detach().numpy()
        predictions = np.argmax(predictions, axis=1)
        yl = yl.detach().numpy()
        f_score = metrics.f1_score(yl, predictions, average='macro')
        tensorboard_writer.add_scalar("F1Score/eval", f_score, epoch)
        test_metrics += f_score
    tqdm.write(f'Epoch {epoch}')
    tqdm.write(f'Eval Epoch loss {epoch_test_loss / len(test_loader)}')
    tqdm.write(f'Eval f1 score {test_metrics / len(test_loader)}')
    return test_metrics / len(test_loader), epoch_test_loss / len(test_loader)

    
def run_training(model, train_loader, test_loader, loss_fn, optimizer, scheduler, early_stop_patience: int = None, EPOCHS=100):
    eval_f1_score = 0.0
    max_loss = 1000
    early_stopping = 0
    writer = SummaryWriter()
    for epoch in range(EPOCHS):
        train_f1, train_loss = train(model, train_loader, loss_fn, optimizer, scheduler, epoch, writer)
        
        test_f1, test_loss = evaluate(model, test_loader, loss_fn, epoch, writer)
        
        if test_f1 > eval_f1_score:
            eval_f1_score = test_f1
            torch.save(model, 'sed_model.pth')
        
        if test_loss < max_loss:
            max_loss = test_loss
            early_stopping = 0
        else:
            print('Early stopping', early_stopping)
            early_stopping += 1
        
        if early_stop_patience and early_stop_patience == early_stopping:
            print('Early stopping stopped')
            break
    return eval_f1_score
            
    
    