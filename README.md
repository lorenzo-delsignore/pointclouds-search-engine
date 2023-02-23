# 3D-PointClouds-ShapeCompletion-and-Classification

The goal of this project is to build a web application, that acts as a search engine for 3D objects, allowing users to interact with a deep learning network and retrieve similar objects.

The application is built with the React.js framework and communicates with the backend using APIs. 

The backend is built using Flask as a microservice, which provides public APIs, like search by query and by object.

# Dataset
The dataset used to train the models is ShapeNet, which is a large-scale dataset of 3D shapes (ref. https://shapenet.org/).

# Dataset loading
Apache Spark was used for data engineering to load the dataset (.parquet files) and perform data exploration and data visualization.

# Models

Contrastive learning is a machine learning technique used to learn representation without supervision.
It can adopt self-defined pseudo labels as supervision and use the learned representations for several downstream tasks.
The key idea of contrastive learning is to embed augmented versions of the same sample close to each other while trying to push away embeddings from different samples.

In the project, the contrastive learning technique is used to the shape completion AutoEncoder.

To evaluate the performance of our approach, we used the PointNet neural network for object classification. 

The training was done with SparkTorch and PyTorch. 

# Conclusion
The experiments show that the use of contrastive learning is slightly similar to the performance the traditional supervised learning methods. (as reference an accuracy of 89.85% in the supervised setting and 89.13% in the constrastive learning setting). 

The purpose of using contrastive learning is to learn the global feature of the point cloud classes, while the Chamfer distance function was used to learn the local features of the shapes, which allowed us to maintain the symmetry and topology of the predicted shapes.
