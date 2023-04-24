import pytest
from knn.preprocessor import Preprocessor

def test_convert_to_float():
    dataset = [['1.0', '2.5'], ['3.5', '4.2']]
    Preprocessor.convert_to_float(dataset, 0)
    assert dataset == [[1.0, '2.5'], [3.5, '4.2']]

def test_convert_to_int():
    dataset = [['apple'], ['banana'], ['apple']]
    mapping = Preprocessor.convert_to_int(dataset, 0)
    assert dataset == [[mapping['apple']], [mapping['banana']], [mapping['apple']]]
    assert set(mapping.values()) == {0, 1} # Ensure the mapping is correct

def test_normalize_dataset():
    dataset = [[10, 20], [15, 30]]
    minmax = [(10, 15), (20, 30)]
    Preprocessor.normalize_dataset(dataset, minmax)
    assert dataset == [[0.0, 0.0], [1.0, 1.0]]

def test_dataset_minmax():
    dataset = [[10, 20], [15, 25], [20, 30]]
    minmax = Preprocessor.dataset_minmax(dataset)
    assert minmax ==  [(10, 20), (20, 30)]

