import pytest
from knn.iris_model import IrisModel

SAMPLE_DATASET_PATH = "tests/data/sample_iris.csv"

@pytest.fixture
def iris_model():
    # Fixture to set up the IrisModel instance
    return IrisModel(filename=SAMPLE_DATASET_PATH, num_neighbors=3, n_folds=3, seed_value=2)

def test_set_seed(iris_model):
    # Test that setting the seed works correctly
    iris_model.set_seed()
    assert iris_model.seed_value == 2 # Ensure the seed is set correctly

def test_load_data(iris_model):
    # Test that data is loaded correctly
    iris_model.load_data()
    assert iris_model.data is not None # Ensure data is loaded
    assert len(iris_model.data) > 0 # Ensure data has rows
    assert isinstance(iris_model.lookup, dict) # Ensure lookup is a dictionary

def test_train_model(iris_model):
    # Test that the model trains correctly
    iris_model.train_model()
    sample_input = [5.1, 3.5, 1.4, 0.2] # Example input
    prediction = iris_model.predict(sample_input)
    assert isinstance(prediction, str) # Ensure the prediciotn is a string
    assert prediction in iris_model.lookup # Ensure the prediction is a valid class

def test_get_scores(iris_model):
    # Test that scores are returned correctly
    iris_model.train_model()
    mean_accuracy, formatted_scores = iris_model.get_scores()
    assert isinstance(mean_accuracy, str) # ensure mean_accuracy is a string
    assert len(formatted_scores) == 3 # Ensure there are as many scores as folds
    assert all(isinstance(score, str) for score in formatted_scores) # Ensure scores are formatted strings



