import logging
from random import randrange
from typing import List, Callable, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CrossValidator:
    """
    A class for evaluating classification algorithms using k-fold cross-validation.

    Attributes:
        n_folds (int): The number of folds to divide the dataset into for cross-validation.
    """

    def __init__(self, n_folds: int) -> None:
        """
        Initialize the CrossValidator with the specified number of folds.

        Args:
            n_folds (int): The number of folds to divide the dataset into for cross-validation.
        """
        self.n_folds = n_folds

    def cross_validation_split(self, dataset: List[List[Any]]) -> List[List[List[Any]]]:
        """
        Split the dataset into k equally sized folds for cross-validation.

        Args:
            dataset (List[List[Any]]): A list of lists representing the dataset.

        Returns:
            List[List[List[Any]]]: A list of folds, where each fold is a list of rows from the dataset.
        """

        logging.info(f"Splitting dataset into (self.n_folds) folds for cross-validation.")

        dataset_split = []
        dataset_copy = list(dataset)
        fold_size = len(dataset) // self.n_folds

        for _ in range(self.n_folds):
            fold = []
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    @staticmethod
    def accuracy_metric(actual: List[Any], predicted: List[Any]) -> float:
        """
        Calculate the classification accuracy of predictions compared to actual values.

        Args:
            actual (List[Any]): A list of actual class labels.
            predicted (List[Any]): A list of predicted class labels.

        Returns:
            float: The percentage of correct predictions (accuracy).
        """

        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        accuracy = correct / len(actual) * 100.0
        logging.info(f"Calculated accuracy: {accuracy:.2f}%")
        return accuracy

    def evaluate_algorithm(self, dataset: List[List[Any]], algorithm: Callable, *args) -> List[float]:
        """
        Evaluate a classification algorithm using k-fold cross-validation.

        Args:
            dataset (List[List[Any]]): A list of lists representing the dataset.
            algorithm (Callable): The classification algorithm to evaluate.
            *args: Additional arguments required by the classification algorithm.

        Returns:
            List[float]: A list of accuracy scores for each fold.
        """

        logging.info("Starting cross-validation evaluation.")
        folds = self.cross_validation_split(dataset)
        scores = []

        for fold in folds:
            train_set = sum((f for f in folds if f is not fold), [])
            test_set = [list(row) for row in fold]
            for row in test_set:
                row[-1] = None # Clear the label for prediciton

            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)

        logging.info(f"Cross-validation completed. Scores: {scores}")
        return scores