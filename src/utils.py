from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.dataset import CLASSES
from src.metrics import get_metric


class TrainEval:
    """
    Class for creating training and evaluating functions in more compact way
    """
    def __init__(self, epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device, model_name,
                 save_dir=None, writer=None):
        """
        :param epochs: Numer of epochs to train model
        :param model: Torch model object
        :param train_dataloader: Train data loader
        :param val_dataloader: Validation data loader
        :param optimizer: Torch optimizer object to use in training
        :param criterion: Torch object to compute loss
        :param device: Torch device object (giving option to use GPU)
        :param model_name: Name of the model, is used when saving model params
        :param writer: tensorboard SummaryWriter
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
        self.writer = writer

        self.save_dir = save_dir or Path(__file__).parent.parent.resolve() / 'models'

        self.save_dir.mkdir(exist_ok=True)

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

            if self.writer:
                self.writer.add_scalar('training loss', loss.item(), current_epoch * len(self.train_dataloader) + t)

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
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for t, data in enumerate(tk):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                all_labels.append(labels.to(torch.device('cpu')).numpy())
                all_logits.append(logits.to(torch.device('cpu')).numpy())

                total_loss += loss.item()
                tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})

                if t == len(tk) - 1:
                    all_labels = np.concatenate(all_labels)
                    all_logits = np.concatenate(all_logits)
                    tk.set_postfix({
                        'Loss': "%6f" % float(total_loss / (t + 1)),
                        'f1': get_metric('f1_score', all_labels, all_logits),
                        'recall': get_metric('recall', all_labels, all_logits),
                        'precision': get_metric('precision', all_labels, all_logits)
                    })

        if self.writer:
            self.writer.add_scalar('validation loss', loss.item(), current_epoch)

            metrics = ['f1_score', 'recall', 'precision']
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

            for metric in metrics:
                for threshold in thresholds:
                    score = get_metric(metric, all_labels, all_logits, threshold=threshold)
                    self.writer.add_scalar(f'{metric} on validation set, threshold = {threshold}', score, current_epoch)

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

        start_time = str(datetime.now()).replace(' ', '_').split('.')[0].replace(':', '-')

        for i in range(self.epoch):
            train_loss = self.train_fn(i)
            val_loss = self.eval_fn(i)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_valid_loss:
                save_path = self.save_dir / f"{self.model_name}_{start_time}_best_weights.pt"
                torch.save(self.model.state_dict(), save_path)
                print("Saved Best Weights")
                best_valid_loss = val_loss
                best_train_loss = train_loss

        save_path = self.save_dir / f"{self.model_name}_{start_time}_last_weights.pt"
        torch.save(self.model.state_dict(), save_path)

        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")

        self.train_losses = train_losses
        self.val_losses = val_losses


def inference(image_path, model, transform=None, threshold=0.5):
    """
    inference on a single image
    :param image_path: path to the image
    :param model: model used for inference
    :param transform: torch.transform applied to the image
    :param threshold: threshold for the prediction
    :return: labels returned by model
    """
    image = Image.open(image_path).convert('RGB')

    if transform is not None:
        image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    output = get_labels(output, threshold)

    return output


def get_labels(predictions, threshold):
    """
    :param predictions: model prediction
    :param threshold: threshold for the prediction
    :return: labels with corresponding probabilities greater than the threshold
    """
    return [[label for label, score in zip(CLASSES, prediction) if score > threshold] for prediction in predictions]