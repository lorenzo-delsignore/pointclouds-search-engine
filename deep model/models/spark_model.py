import torch
from torch import nn

class SparkEncoder(nn.Module):
    def __init__(self, num_points):
        super(SparkEncoder, self).__init__()
        self.num_points = num_points
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
        assert x.shape[1] == self.num_points * 3, "invalid input"

        # convert from fatten to 2048x3 object
        x = x.view(-1, 2048, 3)

        # permute
        x = x.permute(0, 2, 1)

        # encode
        output = self.encoder(x)

        return output.view(-1, output.shape[1])


class SparkDecoder(nn.Module):
    def __init__(self, num_points):
        super(SparkDecoder, self).__init__()
        self.num_points = num_points
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=self.num_points*3),
        )

    def forward(self, x):
        output = self.decoder(x)
        return output.view(-1, self.num_points, 3)

class SparkPointcloudAutoencoder(nn.Module):
    def __init__(self, num_points):
        super(SparkPointcloudAutoencoder, self).__init__()
        self.encoder = SparkEncoder(num_points)
        self.decoder = SparkDecoder(num_points)

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
    