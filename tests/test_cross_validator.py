import pytest
from knn.cross_validator import CrossValidator

@pytest.fixture
def example_dataset():
    # Example dataset: [features..., label]
    return [
        [2.5, 2.4, 0],
        [0.5, 0.7, 0],
        [2.2, 2.9, 0],
        [1.9, 2.2, 1],
        [3.1, 3.0, 1],
        [2.3, 2.7, 0],
        [2, 1.6, 1],
        [1, 1.1, 1],
        [1.5, 1.6, 1],
        [1.1, 0.9, 0]
    ]

def test_cross_validation_split(example_dataset):
    cross_validator = CrossValidator(n_folds=3)
    folds = cross_validator.cross_validation_split(example_dataset)

    assert len(folds) == 3
    assert all(len(fold) > 0 for fold in folds)

def test_accuracy_metric():
    actual = [0,0,1,1,0]
    predicted = [0,1,1,1,0]

    accuracy = CrossValidator.accuracy_metric(actual, predicted)

    assert accuracy == 80.0

def test_evaluate_algorithm(example_dataset):
    cross_validator = CrossValidator(n_folds=3)

    def dummy_algorithm(train, test, *args):
        return [row[-1] for row in test] # simply return the actual labels for testing

    scores = cross_validator.evaluate_algorithm(example_dataset, dummy_algorithm)

    assert len(scores) == 3 # 3 folds
    assert all(0 <= score <= 100 for score in scores)
