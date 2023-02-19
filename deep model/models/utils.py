import os
from pyexpat import features
import random
import shutil
from glob import glob

import gdown
import matplotlib.pyplot as plt
import numpy as np
import torch
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import *
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.functions import (PandasUDFType, col, count,
                                   monotonically_increasing_id, pandas_udf,
                                   rand, udf)
from torch.utils.data import DataLoader, Dataset
from models.data import PointCloudData
from models.datautil import load_obj, pcshow
from models.autoencoder import PointcloudAutoencoder
from models.pointnet import PointNet
from models.pointsampler import PointSampler
from models.transformation import transform_input
import json
import pyarrow as pa
import pyarrow.parquet as pq

def save_autoencoder_state(model, num_classes, scheduler, optimizer, log_dict):
    torch.save(model.state_dict(), f"state/autoencoder_model_{num_classes}c.pt")
    torch.save(scheduler.state_dict(), f"state/autoencoder_scheduler_{num_classes}c.pt")
    torch.save(optimizer.state_dict(), f"state/autoencoder_optimizer_{num_classes}c.pt")

    with open(f"state/autoencoder_log_dict_{num_classes}c.json", "w") as f:
        json.dump(log_dict, f)

def load_autoencoder_state(model, num_classes, contrastive, scheduler=None, optimizer=None, device=None):
    contrastive_prefix = "_contrastive" if contrastive else ""
    model.load_state_dict(torch.load(f"state/autoencoder_model_{num_classes}c{contrastive_prefix}.pt", map_location=None if device is None else torch.device(device)))

    if scheduler is not None:
        scheduler.load_state_dict(torch.load(f"state/autoencoder_scheduler_{num_classes}c{contrastive_prefix}.pt"))

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(f"state/autoencoder_optimizer_{num_classes}c{contrastive_prefix}.pt"))

    with open(f"state/autoencoder_log_dict_{num_classes}c{contrastive_prefix}.json", "r+") as f:
        log_dict = json.load(f)

    return log_dict

def load_pointnet_state(model, num_classes, optimizer=None, device=None):
    model.load_state_dict(torch.load(f"state/pointnet_model_{num_classes}c.pt", map_location=None if device is None else torch.device(device)))

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(f"state/poitnet_optimizer_{num_classes}c.pt"))

    with open(f"state/log_dict_pointnet_{num_classes}c.json", "r+") as f:
        log_dict = json.load(f)

    return log_dict

def get_class_from_object(obj):
    device = "cpu" 
    num_classes = 10

    # load autoencoder state
    autoencoder = PointcloudAutoencoder(2048)
    autoencoder.to(device)
    load_autoencoder_state(autoencoder, num_classes=num_classes, contrastive=True, device=device)
    autoencoder.eval()

    # load pointcloud
    verts, faces = load_obj(obj)
    pointcloud = PointSampler(2048)((verts, faces))
    pointcloud = transform_input(pointcloud)

    # get autoencoder result
    features = pointcloud.unsqueeze(dim=0).to(device).float()
    decoded_obj = autoencoder(features.permute(0, 2, 1))

    # load del classificatore
    pointnet = PointNet(num_classes)
    pointnet.to(device)
    load_pointnet_state(pointnet, num_classes=num_classes, device=device)
    pointnet.eval()

    # find classes
    predicted_classes, _, _ = pointnet(decoded_obj.permute(0, 2, 1))
    print(predicted_classes)
    _, predicted_classes = torch.max(predicted_classes.data, 1)
    predicted_class = predicted_classes.item()

    class_dict= {
        0: 'airplane',
        1: 'car',
        2: 'chair',
        3: 'table',
        4: 'bench', 
        5: 'lamp', 
        6: 'sofa' ,
        7: 'rifle', 
        8: 'speaker', 
        9: 'vessel'
    }

    return class_dict[predicted_class]

def create_spark_session(spark_num_cores):
    # how many rows per batch e.g. in @pandas_udf functions  
    SPARK_MAX_RECORDS_PER_BATCH = 1e3
    # how many bytes can a single partition store
    SPARK_MAX_PARTITION_BYTES = 1e8
    
    # create spark 
    spark = SparkSession.builder \
        .config("spark.ui.port", "4050") \
        .config('spark.executor.memory', '10G') \
        .config('spark.driver.memory', '10G') \
        .config('spark.driver.maxResultSize', '10G') \
        .config("spark.sql.execution.arrow.pyspark.enabled", True) \
        .config("spark.memory.offHeap.enabled", True) \
        .config("spark.memory.offHeap.size","16g") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", int(SPARK_MAX_RECORDS_PER_BATCH)) \
        .config("spark.sql.files.maxPartitionBytes", int(SPARK_MAX_PARTITION_BYTES)) \
        .master("local[{}]".format(spark_num_cores)) \
        .getOrCreate()

    return spark

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def download_dataset(output_folder="."):
    # create the folder if doesn't exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    is_dataset_complete = True
    dataset_data = [ "part_0000.parquet", "part_0001.parquet", "part_0002.parquet", "part_0003.parquet", "part_0004.parquet", "part_0005.parquet" ]
    
    for part_id in dataset_data:
        part_path = os.path.join(output_folder, part_id)

        if not os.path.exists(part_path):
            is_dataset_complete = False
            break
            
    if not is_dataset_complete:
        gdown.download_folder("https://drive.google.com/drive/folders/1cp0ej6wcoG-2wXMJ2RoaU1fzb3CiG_te", output=output_folder, quiet=False)

    return glob(os.path.join(output_folder, "*.parquet"))

def undersample(df):
    result = None
    
    for split in [ 'train', 'test', 'val' ]:
        data = df[df['split'] == split].groupBy("label").count().collect()
        mean = int(np.mean([ x['count'] for x in data ]))

        fractions = {}
        
        for x in data:
            fractions[x['label']] = np.min([1.0, mean / x['count']])

        sampledByLabel = df[df['split'] == split].sampleBy("label", fractions=fractions, seed=42)
        result = sampledByLabel if result is None else result.union(sampledByLabel)

    return result

def get_dataset(spark):
    df = spark.read.parquet(*glob(os.path.join("data", "*.parquet")))

    # filter by label
    filter_by_label = (df['label'] == 'airplane') | (df['label'] == 'car') | (df['label'] == 'chair') | (df['label'] == 'table') | (df['label'] == 'bench') | (df['label'] == 'lamp') | (df['label'] == 'sofa') | (df['label'] == 'rifle') | (df['label'] == 'speaker') | (df['label'] == 'vessel') 
    df = df[filter_by_label]

    window = Window.orderBy("split").partitionBy("split") 
    df = df.withColumn("index", row_number().over(window) - 1) 
    df.cache()
   
    return df

def show_dataset(dataset, n_classes=10):
    # take one sample per class
    samples = []
    labels = []

    for (features, id) in dataset:
        label = dataset.id2label[id]

        if not label in labels:
            labels.append(label)
            samples.append(features)

        if len(labels) == n_classes: break

    samples = np.array([ np.array(sample).reshape(2048, 3) for sample in samples ])
    fig = plt.figure(figsize=(20, 5))
    
    for i in range(n_classes):
        ax = fig.add_subplot(2, n_classes, i + 1, projection='3d')
        ax.scatter(samples[i,:,0], samples[i,:,2], samples[i,:,1], c='b', marker='.', alpha=0.8, s=1)
        ax.set_title(label=labels[i])
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    