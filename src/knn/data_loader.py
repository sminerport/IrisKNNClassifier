from csv import reader
from typing import List

class DataLoader:
    """
    A class for loading and preprocessing a dataset from a CSV file.

    Attributes:
        filename (str): The path to the CSV file.
        skip_header (bool): Whether to skip the header row.

    Methods:
        load_data(): Load the CSV file and return its contents as a list of lists, optionally skipping the header row.
    """

    def __init__(self, filename: str, skip_header=False):
        """
        Initialize a DataLoader instance.

        Args:
            filename (str): The path to the CSV file.
            skp_header (bool): Whether to skip the header row.
        """
        self.filename = filename
        self.skip_header = skip_header

    def load_data(self):
        """
        Load the CSV file and return its contents as a list of lists.

        Depending on the value of skip_header, the method will either include or exclude
        the first row (header) of the CSV file.

        Returns:
            list: A list of lists containing the rows of the CSV file.
        """

        dataset = []

        with open(self.filename, "r") as file:
            csv_reader = reader(file)
            if self.skip_header:
                next(csv_reader, None)  # Skip the header row
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset
