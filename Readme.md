# NeuralVision

## Description

NeuralVision is a project that involves developing a neural network to classify images into various categories. It demonstrates proficiency in machine learning, particularly in the implementation and training of convolutional neural networks (CNNs) using Python and relevant libraries.

## Table of Contents
- [Description](#description)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Classes and Methods](#functions)
- [Outcomes](#outcomes)
- [License](#license)

## Technologies Used

- Python 3.x
- TensorFlow
- scikit-learn
- OpenCV

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jasmeets7/neuralvision.git
    ```
2. Navigate to the project directory:
    ```bash
    cd neuralvision
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the image classification model, execute the `main.py` script:
```sh
python main.py data_directory [model.h5]
```
- `data_directory`: Path to the directory containing image data organized in subdirectories by category.
- `[model.h5]`: (Optional) Filename to save the trained model.

## Project Structure
- `main.py`: Main script to run the neural network model, including the main execution logic.
- `data_preprocessing.py`: Contains the `DataLoader` class for loading and preprocessing of the data.
- `model.py`: Contains the `CNNModel(Convolutional Neural Network Model)` class to define the model architecture.
- `requirements.txt`: Contains the list of packages required for the project.

## Functions

### data_preprocessing.py
- **DataLoader**
    - **load_data(data_dir)**: Loads and preprocesses images from the specified data directory, resizing them to the required dimensions and returning the images and their corresponding labels.

### model.py
- **CNNModel(Convolutional Neural Network Model)**
    - **build_model()**: Constructs the CNN architecture.
    - **get_model()**: Returns the compiled CNN model.

### main.py
- **main()**: Main function to load data, split it into training and testing sets, build and train the model, evaluate the model, and optionally save the trained model.

## Outcomes
- Implemented and trained a convolutional neural network (CNN) for image classification.
- Applied data preprocessing techniques using OpenCV to prepare images for training.
- Utilized Python libraries such as TensorFlow and scikit-learn to build, train, and evaluate machine learning models.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.