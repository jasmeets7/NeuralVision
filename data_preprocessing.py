from cv2 import imread, resize
from os import listdir, path

class DataLoader:
    def __init__(self, img_width=30, img_height=30):
        self.img_width = img_width
        self.img_height = img_height

    def load_data(self, data_dir):
        directories = listdir(data_dir)
        images, labels = [], []
        for directory in directories:
            for file in listdir(path.join(data_dir, directory)):
                file_path = path.join(data_dir, directory, file)
                image = imread(file_path)
                resized_image = resize(image, (self.img_width, self.img_height))

                images.append(resized_image)
                labels.append(directory)
        return images, labels