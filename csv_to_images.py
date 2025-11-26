import pandas as pd
import numpy as np
from PIL import Image
import os

# Paths to your CSV files
train_csv = 'dataset/raw/sign_mnist_train.csv'
test_csv = 'dataset/raw/sign_mnist_test.csv'

# Output folders inside raw/
train_output = 'dataset/raw/images/train'
test_output = 'dataset/raw/images/test'

# Function to convert CSV to images
def csv_to_images(csv_file, output_folder):
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        label = row[0]  # first column is the label
        pixels = row[1:].values.astype(np.uint8).reshape(28, 28)  # flatten to 28x28

        # Create folder for each label
        label_folder = os.path.join(output_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)

        # Save image
        img = Image.fromarray(pixels)
        img.save(os.path.join(label_folder, f'{index}.png'))

# Convert CSV to images
csv_to_images(train_csv, train_output)
csv_to_images(test_csv, test_output)

print("CSV converted to images inside dataset/raw/images successfully!")
