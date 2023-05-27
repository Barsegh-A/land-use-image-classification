import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


class TrainEval:
    """
    Class for creating training and evaluating functions in more compact way
    """
    def __init__(self, epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device, model_name):
        """
        :param epochs: Numer of epochs to train model
        :param model: Torch model object
        :param train_dataloader: Train data loader
        :param val_dataloader: Validation data loader
        :param optimizer: Torch optimizer object to use in training
        :param criterion: Torch object to compute loss
        :param device: Torch device object (giving option to use GPU)
        :param model_name: Name of the model, is used when saving model params
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epochs
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []

    def train_fn(self, current_epoch):
        """
        One epoch model training
        :param current_epoch: Number indicating epoch
        :return: Epoch mean loss
        """
        self.model.train()
        self.model.to(self.device)
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch + 1) + "/" + str(self.epoch))

        for t, data in enumerate(tk):
            images, labels = data

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})

        return total_loss / len(self.train_dataloader)

    def eval_fn(self, current_epoch):
        """
        Model validation function
        :param current_epoch: Number indicating epoch
        :return: Mean validation loss
        """
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        tk = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(self.epoch))

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})

        return total_loss / len(self.val_dataloader)

    def train(self):
        """
        Runs training on model and saves weights of the model having the best validation loss
        :return:
        """
        best_valid_loss = np.inf
        best_train_loss = np.inf

        train_losses = []
        val_losses = []

        for i in range(self.epoch):
            train_loss = self.train_fn(i)
            val_loss = self.eval_fn(i)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_valid_loss:
                save_path = Path(__file__).parent.parent.resolve() / 'models' / f"{self.model_name}_best_weights.pt"
                torch.save(self.model.state_dict(), save_path)
                print("Saved Best Weights")
                best_valid_loss = val_loss
                best_train_loss = train_loss
        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")

        self.train_losses = train_losses
        self.val_losses = val_losses