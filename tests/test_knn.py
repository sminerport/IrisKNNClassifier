import pytest
from knn.k_nearest_neighbors import KNearestNeighbors

def test_euclidean_distance():
    knn = KNearestNeighbors(num_neighbors=3)
    row1 = [2.0, 3.0, 0]
    row2 = [5.0, 7.0, 1]
    expected_distance = 5.0 # sqrt((2-5^2 + (3-7^2) = sqrt(9 + 16) = sqrt(25) = 5
    assert knn.euclidean_distance(row1, row2) == expected_distance

def test_get_neighbors():
    knn = KNearestNeighbors(num_neighbors=3)
    train = [
        [1.0, 2.0, 0],
        [2.0, 3.0, 0],
        [3.0, 4.0, 1],
        [6.0, 7.0, 1]
    ]
    test_row = [2.5, 3.5, None]
    expected_neighbors = [
        [2.0, 3.0, 0],
        [3.0, 4.0, 1],
        [1.0, 2.0, 0]
    ]
    neighbors = knn.get_neighbors(train, test_row)
    assert neighbors == expected_neighbors

def test_predict_classification():
    knn = KNearestNeighbors(num_neighbors=3)
    train = [
        [1.0, 2.0, 0],
        [2.0, 3.0, 0],
        [3.0, 4.0, 1],
        [6.0, 7.0, 1]
    ]
    test_row = [2.5, 3.5, None]
    expected_prediction = 0 # Neighbors [0, 0, 1] -> most common class is 0
    prediction = knn.predict_classification(train, test_row)
    assert prediction == expected_prediction

def test_predict_all():
    knn = KNearestNeighbors(num_neighbors=3)
    train = [
        [1.0, 2.0, 0],
        [2.0, 3.0, 0],
        [3.0, 4.0, 1],
        [6.0, 7.0, 1]
    ]
    test = [
        [2.5, 3.5, None],
        [5.0, 6.0, None]
    ]
    expected_predictions = [0, 1] # First row predicts 0, second row predicts 1
    predictions = knn.predict_all(train, test)
    assert predictions == expected_predictions