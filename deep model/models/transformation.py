import math
import numpy as np
import random
import torch
from models.pointsampler import PointSampler
from torchvision import datasets, models, transforms

class FromFlattenToPointcloud(object):
     def __call__(self, pointcloud):
        assert len(pointcloud.shape)==1
        return pointcloud.reshape(2048, 3)

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
        # PointSampler(2048),
        # Normalize(),
        ToTensor()
    ])

def transform_input(pointcloud):
    return transforms.Compose([
        Normalize(),
        ToTensor()
    ])(pointcloud)

def pointnet_default_transform():
    return transforms.Compose([
        FromFlattenToPointcloud(),
        Normalize(),
        ToTensor()
    ])

def pointnet_train_transforms():
    return transforms.Compose([
        FromFlattenToPointcloud(),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])
