# Iris Flowers Classification with k-Nearest Neighbors

This repository contains a Python script that implements the k-Nearest Neighbors (k-NN) algorithm for classifying Iris flowers into three species: setosa, versicolor, and virginica. The script uses the Iris dataset, which consists of 150 samples with 4 features each: sepal length, sepal width, petal length, and petal width.

## Getting Started

These instructions will guide you on how to run the script on your local machine.

### Prerequisites

To run this script, you'll need Python 3 and the following libraries installed:

- pandas
- scikit-learn

You can install them using pip:

```sh
pip install pandas scikit-learn
```

### Running the Script

1. Clone this repository to your local machine:

```sh
git clone https://github.com/sminerport/iris-knn-classifier.git
```

2. Navigate to the repository's directory:

```sh
cd iris-classification-knn
```

3. Run the script:

```sh
python iris_knn.py
```

The script will train the k-NN model on the Iris dataset, using 5-fold cross-validation and 10 neighbors (you can change these values). It will print the accuracy for each fold and the mean accuracy.

After training the model, you can input your own values for sepal length, sepal width, petal length, and petal width. The model will predict the flower category (setosa, versicolor, or virginica) based on the input provided by the user.

## Customizing the Script

You can customize the number of cross-validation folds and the number of neighbors in the k-NN algorithm by modifying the n_folds and num_neighbors variables in the script.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* The Iris dataset was introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems."
* The k-Nearest Neighbors algorithm is a simple yet powerful classification technique, particularly suitable for problems with small datasets and relatively low-dimensional feature spaces.
