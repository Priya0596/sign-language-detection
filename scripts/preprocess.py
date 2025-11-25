import pandas as pd
import numpy as np
import os

def load_and_preprocess(csv_path):
    data = pd.read_csv(csv_path)
    labels = data['label'].values
    images = data.drop('label', axis=1).values
    images = images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return images, labels

if __name__ == "__main__":
    train_csv = "./dataset/raw/sign_mnist_train.csv"
    test_csv  = "./dataset/raw/sign_mnist_test.csv"

    X_train, y_train = load_and_preprocess(train_csv)
    X_test, y_test = load_and_preprocess(test_csv)

    np.save("./dataset/processed/X_train.npy", X_train)
    np.save("./dataset/processed/y_train.npy", y_train)
    np.save("./dataset/processed/X_test.npy", X_test)
    np.save("./dataset/processed/y_test.npy", y_test)

    print("Preprocessing completed!")
