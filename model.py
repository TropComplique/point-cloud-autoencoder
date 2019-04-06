import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, k, num_points):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super(Autoencoder, self).__init__()

        # ENCODER

        pointwise_layers = []
        num_units = [3, 64, 128, 128, 256, k]

        for n, m in zip(num_units[:-1], num_units[1:]):
            pointwise_layers.extend([
                nn.Conv1d(n, m, kernel_size=1, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True)
            ])

        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        # DECODER

        self.decoder = nn.Sequential(
            nn.Conv1d(k, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """

        b, _, num_points = x.size()
        x = self.pointwise_layers(x)  # shape [b, k, num_points]
        encoding = self.pooling(x)  # shape [b, k, 1]

        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, 3, num_points)

        return encoding, restoration
