from math import sqrt
from typing import List, Union, Tuple
import heapq

class KNearestNeighbors:
    """
    A class for the k-nearest neighbors classification algorithm.
    """

    def __init__(self, num_neighbors: int) -> None:
        """
        Initialize the KNearestNeighbors with the specified number of neighbors.

        Args:
            num_neighbors (int): The number of neighbors to use for the predictions.
        """
        self.num_neighbors = num_neighbors

    @staticmethod
    def euclidean_distance(row1: List[float], row2: List[float]) -> float:
        """
        Calculate the Euclidean distance between two rows.

        Arts:
            row1 (List[float]): The first row containing numerical values.
            row2 (List[float]): The second row containing numerical values.

        Returns:
            float: The Euclidean distance between the two rows.
        """
        return sqrt(sum((r1 - r2) ** 2 for r1, r2 in zip(row1[:-1], row2[:-1])));

    def get_neighbors(self, train: List[List[float]], test_row: List[float]) -> List[List[float]]:
        """
        Locate the k most similar neighbors in the training set for a given test row.

        Args:
            train (List[List[float]]): A list of lists representing the training dataset.
            test_row (List[float]): A row from the test dataset containing numerical values.

        Returns:
            List[List[float]]: A list of the k nearest neighbors from the training dataset.
        """

        distances = [(self.euclidean_distance(test_row, train_row), train_row) for train_row in train]

        # Use heapq to get the k smallest elements
        neighbors = heapq.nsmallest(self.num_neighbors, distances, key=lambda x: x[0])
        return [neighbor[1] for neighbor in neighbors]

    def predict_classification(self, train: List[List[float]], test_row: List[float]) -> Union[int, str]:
        """
        Make a classification prediction for a test row using k-nearest neighbors.

        Args:
            train (List[List[float]]): A list of lists representing the training dataset.
            test_row (List[float]): A row from the test dataset containing numerical values.

        Returns:
            Union[int, str]: The predicted class label for the test row.
        """

        neighbors = self.get_neighbors(train, test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    def predict_all(self, train: List[List[float]], test: List[List[float]]) -> List[Union[int, str]]:
        """
        Make classification predictions for all rows in the test dataset using the k-nearest neighbors algorithm.

        Args:
            train (List[List[float]]): A list of lists representing the training dataset.
            test (List[List[float]]): A list of lists representing the test dataset.

        Returns:
            List[Union[int, str]]: A list of predicted class labels for all rows in the test dataset.
        """

        return [self.predict_classification(train, row) for row in test]
