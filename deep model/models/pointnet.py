
import torch
from torch import nn

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()     
        self.k=k
        self.my_network = torch.nn.Sequential(
            torch.nn.Conv1d(k, 64, 1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 1024, 1, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2048),
            torch.nn.Flatten(1),
            torch.nn.Linear(1024, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256, bias=False),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, k*k)
        )
        
    def forward(self, input):
        bs = input.size(0)
        #initialize as identity
        device = "cuda:0" if next(self.parameters()).is_cuda else "cpu"
        init = torch.eye(self.k, requires_grad=True, device=device).repeat(bs,1,1)
        matrix = self.my_network(input).view(-1, self.k, self.k) + init
        return matrix

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.first_block = torch.nn.Sequential(
            torch.nn.Conv1d(3, 64, 1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.second_block = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 1, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 1024, 1, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.MaxPool1d(2048),
            torch.nn.Flatten(1)
        )

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        xb = self.first_block(xb)
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)
        output = self.second_block(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.my_network = torch.nn.Sequential(
            torch.nn.Linear(1024, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256, bias=False),
            torch.nn.Dropout(p=0.3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, classes),
            torch.nn.LogSoftmax(dim=1)
        )

    def check_input(self, input):
        # convert from fatten to 2048x3 object
        if input.shape[1] == 6144:
            input = input.view(-1, 2048, 3)

            # permute
            input = input.permute(0, 2, 1)

        return input

    def forward(self, input):
        input = self.check_input(input)
        xb, matrix3x3, matrix64x64 = self.transform(input)
        output = self.my_network(xb)
        return output, matrix3x3, matrix64x64
