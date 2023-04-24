import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import re

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
     # create dictionary
	for i, value in enumerate(unique):
		lookup[value] = i
     # convert dataset column
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
          # create hold out set
		train_set.remove(fold)
          #combine train sets
		train_set = sum(train_set, [])
          # create test set on new hold
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
               # remove prediction from hold out set
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1) - 1):
		distance += (row1[i] - row2[i]) ** 2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

# Test the kNN on the Iris Flowers dataset
seed(1)
filename = 'data/iris.txt'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
	str_column_to_float(dataset[1:], i)
# convert class column to integers
# versicolor : 0
# virginica: 1
# setosa: 2
lookup = str_column_to_int(dataset[1:], len(dataset[0]) - 1)

# evaluate algorithm
n_folds = 5
num_neighbors = 10
scores = evaluate_algorithm(dataset[1:], k_nearest_neighbors, n_folds, num_neighbors)
print(f'****************************************************************************************')
print(f'*')                                                                                      
print(f'*   K-Nearest Neighbor (KNN) algorithm with {num_neighbors} neighbors trained on a ')
print(f'*   dataset containing  {len(dataset)-1} rows and {len(dataset[1])-1} features, using {n_folds}-fold cross validation.')   
print(f'*')                                        
print(f'*   Users can adjust the n_folds and num_neighbors variables in the script.')
print(f'*')                                        
print(f'****************************************************************************************')
print()
print(f'Accuracy per fold: {scores}')
print(f'Mean Accuracy: {sum(scores) / float(len(scores)):.3f}')

while True:
    try:
        sl, sw, pl, pw = [float(x) for x in re.split(r'\, | |\,', input('\nPlease input four floating point numbers, representing, respectively,\n\
sepal length, sepal width, petal length, and petal width \n(e.g., 5.1, 3.5, 1.4, 0.2).  The model will guess the flower category\n\
(i.e., setosa, versicolor, or virginica) based on your input (CTRL-C to Exit): '))]
        test_row = [sl, sw, pl, pw]
        test_row
        prediction = predict_classification(train=dataset[1:], test_row=test_row, num_neighbors=num_neighbors)
        for key, value in lookup.items():
               
               if prediction == value:
                   prediction = key
        print()
        print(f'Prediction: {prediction}')
    except ValueError:
        print('\nNote: wrong input format.')
