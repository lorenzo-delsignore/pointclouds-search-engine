import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.AdaptiveMaxPool1d(output_size=1)
        )

    def forward(self, x):
        output = self.encoder(x)
        return output.view(-1, output.shape[1])

class Decoder(nn.Module):
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=6144),
        )

    def forward(self, x):
        output = self.decoder(x)
        return output.view(-1, self.num_points, 3)

class PointcloudAutoencoder(nn.Module):
    def __init__(self, num_points):
        super(PointcloudAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)

    def forward(self, pointclouds):
        gfv = self.encoder(pointclouds)
        out = self.decoder(gfv)
        return out

    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        return self.encoder(pointclouds)

    def reconstruct(self, pointclouds):
        return self.decoder(pointclouds)
