from email.policy import default
import os
import csv
import pandas as pd
from collections import defaultdict
from pointsampler import PointSampler

input_file = "all.csv"
output_file = "split.csv"
dataset_path = "../../ShapeNetCore.v2"

id2cat = {
    '02691156': 'airplane',
    '02958343': 'car',
    '03001627': 'chair',
    '04379243': 'table',
    '02828884': 'bench', 
    '03636649': 'lamp', 
    '04256520': 'sofa' ,
    '04090263': 'rifle', 
    '03691459': 'speaker', 
    '04530566': 'vessel'
}

classes = id2cat.keys()

def write_csv(filename, headers, data):
    with open(filename, 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # write the header
        writer.writerow(headers)

        # write the data
        writer.writerows(data)

def describe_csv(filename):
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter=",")

        val = []
        test = []
        train = []
        headers = next(data)
        test_samples_per_class = defaultdict(lambda: 0)
        training_samples_per_class = defaultdict(lambda: 0)
        validation_samples_per_class = defaultdict(lambda: 0)

        for row in data:
            row_type = row[4]
            synsetId = row[1]

            if not synsetId in classes:
                continue
            
            if row_type == 'train':
                train.append(row)
                training_samples_per_class[synsetId] += 1
            elif row_type == 'test':
                test.append(row)
                test_samples_per_class[synsetId] += 1
            elif row_type == 'val':
                val.append(row)
                validation_samples_per_class[synsetId] += 1

        print(f"The number of training samples are {len(train)}")
        print(f"The number of valadation samples are {len(val)}")
        print(f"The number of testing samples are {len(test)}")

        print(f"Here there is the training samples per class, average: {sum(training_samples_per_class.values())/len(training_samples_per_class):.0f}.")
        print(dict(training_samples_per_class))

        print(f"Here there is the validation samples per class, average: {sum(validation_samples_per_class.values())/len(validation_samples_per_class):.0f}.")
        print(dict(validation_samples_per_class))

        print(f"Here there is the testing samples per class, average: {sum(test_samples_per_class.values())/len(test_samples_per_class):.0f}.")
        print(dict(test_samples_per_class))

def balance_dataset(input_file, output_file, max_training_samples_per_class=None, max_testing_samples_per_class=None, max_val_samples_per_class=None, dataset_path=dataset_path):
    if not os.path.exists(dataset_path):
        raise "Dataset not found."

    output = []
    samples_per_class = defaultdict(lambda: defaultdict(lambda: 0))
    max_samples_per_class = {
        'train':max_training_samples_per_class,
        'test':max_testing_samples_per_class,
        'val':max_val_samples_per_class
    }

    with open(input_file) as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        headers = next(data)

        for row in data:
            row_type = row[4]
            synsetId = row[1]

            if not synsetId in classes:
                continue

            model_path = os.path.join(dataset_path, f"{synsetId}/{row[3]}/models/model_normalized.obj")

            if not os.path.exists(model_path):
                continue

            samples_per_class[row_type][synsetId] += 1
            write_into_csv = row_type in max_samples_per_class and (max_samples_per_class[row_type] is None or samples_per_class[row_type][synsetId] <= max_samples_per_class[row_type])

            if write_into_csv:
                output.append(row)

        write_csv(output_file, headers, output)

def get_model_pointcloud(model_path):
    verts = []
    faces = []

    with open(model_path, "r", encoding="utf-8") as file:
        data = file.read()

        for line in data.split("\n"):
            if line == "":
                    continue

            elif line.lstrip().startswith("v "):
                vertices = line.replace("\n", "").split(" ")[1:]
                verts.append(list(map(float, vertices)))

            elif line.lstrip().startswith("f "):
                t_index_list = []
                for t in line.replace("\n", "").split(" ")[2:]:
                    t_index = t.split("/")[0]
                    t_index_list.append(int(t_index) - 1)
                faces.append(t_index_list)

        pointcloud = PointSampler(2048)((verts, faces))
        return pointcloud

def df_to_parquet(df, target_dir, chunk_size=1000000, **parquet_wargs):
    for i in range(0, len(df), chunk_size):
        slc = df.iloc[i : i + chunk_size]
        chunk = int(i/chunk_size)
        fname = os.path.join(target_dir, f"part_{chunk:04d}.parquet")
        slc.to_parquet(fname, engine="pyarrow", **parquet_wargs)

def csv_into_parquet(filename, target_dir, dataset_path=dataset_path):
    with open(filename, encoding="utf-8") as csvfile:
        data = csv.reader(csvfile, delimiter=",")
        headers = next(data)
        
        all_data = []
        all_label = []
        all_split = []

        for row in data:
            id, synsetId, subSynsetId, modelId, split = row
            model_path = os.path.join(dataset_path, f"{synsetId}/{modelId}/models/model_normalized.obj")
            pointcloud = get_model_pointcloud(model_path)
            all_label.append(id2cat[synsetId])
            all_data.append(pointcloud)
            all_split.append(split)

        df_columns = [ 'features', 'label', 'split']
        df_data = {         
            "features": [ sample.flatten() for sample in all_data ], 
            "label": all_label, 
            "split": all_split
        }

        df = pd.DataFrame(df_data, columns=df_columns)
        df_to_parquet(df, target_dir, chunk_size=10000)

if __name__ == "__main__":
    # describe input dataset
    describe_csv(input_file)           

    # balance training set
    balance_dataset(input_file, output_file, max_training_samples_per_class=3000, max_testing_samples_per_class=360, max_val_samples_per_class=715)

    # describe balanced dataset
    describe_csv(output_file)

    # convert the balanaced dataset in parquet file
    csv_into_parquet(output_file, 'data')
