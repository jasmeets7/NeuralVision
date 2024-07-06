from sys import exit, argv

from data_preprocessing import DataLoader
from model import CNNModel

from numpy import array
from sklearn.model_selection import train_test_split
import tensorflow as tf

EPOCHS = 10
TEST_SIZE = 0.4

def main():

    if len(argv) not in [2, 3]:
        exit("Usage: python main.py data_directory [model.h5]")

    data_loader = DataLoader()
    images, labels = data_loader.load_data(argv[1])

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        array(images), array(labels), test_size=TEST_SIZE
    )

    cnn_model = CNNModel()
    model = cnn_model.get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    if len(argv) == 3:
        filename = argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

if __name__ == "__main__":
    main()