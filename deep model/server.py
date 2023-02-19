import os
import uuid
import numpy as np

from flask_cors import CORS
from markupsafe import escape
from flask import Flask, request, abort
from werkzeug.utils import secure_filename

from models.utils import create_spark_session, get_dataset, get_class_from_object

maxImagesPerPage = 15

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# create spark session
spark = create_spark_session(4)
df = get_dataset(spark)
df = df.toPandas()

@app.route("/api/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print('No file part')
        return abort(400)

    file = request.files['file']

    if not file:
        return abort(400)

    filename, file_extension = os.path.splitext(secure_filename(file.filename))

    if not filename or file_extension != ".obj":
        return abort(400)

    id =  str(uuid.uuid1())

    # upload file
    temp_file =  os.path.join(app.config['UPLOAD_FOLDER'], id + file_extension)
    file.save(temp_file)

    # return response
    return { "id": id }

@app.route("/api/available-classes", methods=['GET'])
def get_available_classes():
    return [ row for row in df["label"].unique() ]

@app.route("/api/search-by-query/<value>", methods=['GET'])
def search_by_keyword(value):
    value = escape(value)

    if value:
        value = value.lower()

    # in caso di keywords multiple, se ci sono più classi, mostriamo solo quelle disponibili.
    any_valid_class = any(value.find(available_class) != -1 for available_class in get_available_classes())

    if not any_valid_class:
        return {
            "input": value, # invece di inviare l'oggetto, inviare il link all'oggetto.
            "similarImages": []
        }

    query_images = value.split(" ") 
    images = []

    # aggregate images by query
    for query_image in query_images:
        query_result = df[df['label'] == query_image].sample(maxImagesPerPage)

        if query_result is not None:
            images.extend([ 
                { "image": np.array(row['features']).reshape(2048, 3).tolist() } for idx, row in query_result.iterrows()
             ])

    # build response
    result = {
        "page": 1,
        "keywords": value,
        "similarImages": images[:maxImagesPerPage]
    }

    return result

@app.route("/api/search-by-object/<value>", methods=['GET'])
def search_by_object(value):
    temp_file  = os.path.join(app.config['UPLOAD_FOLDER'], value + ".obj")
    predicted_class = get_class_from_object(temp_file)
    query_result = df[df['label'] == predicted_class].sample(maxImagesPerPage)
    images = [{"image": np.array(row['features']).reshape(2048, 3).tolist() } for idx, row in query_result.iterrows() ]

    result = {
        "page": 1,
        "predicted_class": predicted_class,
        "similarImages": images[:maxImagesPerPage]
    }

    return result
    