from random import seed
from knn.data_loader import DataLoader
from knn.preprocessor import Preprocessor
from knn.cross_validator import CrossValidator
from knn.k_nearest_neighbors import KNearestNeighbors

class IrisModel:
    def __init__(self, filename, num_neighbors=10, n_folds=5, seed_value=2):
        self.filename = filename
        self.num_neighbors = num_neighbors
        self.n_folds = n_folds
        self.knn = None
        self.data = None
        self.lookup = None
        self.mean_accuracy = None
        self.scores = None
        self.seed_value = seed_value
        self.set_seed()
        self.load_data()
        self.train_model()

    def set_seed(self):
        seed(self.seed_value)

    def load_data(self):
        dl = DataLoader(self.filename)
        dataset = dl.load_data()

        # Split header and data
        self.header, self.data = dataset[0], dataset[1:]

        # Convert columns to float and labels to integers
        for column_index in range(len(self.header) - 1):
            Preprocessor.convert_to_float(self.data, column_index)

        self.lookup = Preprocessor.convert_to_int(self.data, len(self.header) - 1)

    def train_model(self):
        cross_validator = CrossValidator(n_folds=self.n_folds)
        self.knn = KNearestNeighbors(self.num_neighbors)
        self.scores = cross_validator.evaluate_algorithm(self.data, self.knn.predict_all)
        self.mean_accuracy = sum(self.scores) / float(len(self.scores))

    def predict(self, inputs):
        prediction = self.knn.predict_classification(self.data, inputs)
        for key, value in self.lookup.items():
            if prediction == value:
                return key
        return prediction

    def get_scores(self):
        formatted_scores = [f"{score:.2f}" for score in self.scores]
        formatted_mean_accuracy = f"{self.mean_accuracy:.2f}"
        return formatted_mean_accuracy, formatted_scores

