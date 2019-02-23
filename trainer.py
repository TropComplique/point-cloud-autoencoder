import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import Autoencoder
from loss import ChamferDistance


class Trainer:
    def __init__(self, num_steps):
        """
        """
        self.network = Autoencoder()
        self.loss = ChamferDistance()

        self.optimizer = optim.Adam(lr=1e-3, params=self.network.parameters(), weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=1e-7)

    def train_step(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            a float tensor with shape [].
        """

        encoding, restoration = self.network(x)

        loss = self.loss(x, restoration)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def evaluate(self, images, labels):
        """
        Evaluation is on batches of size 1.
        """

        with torch.no_grad():
            losses = self.get_losses(images, labels)

        return losses
