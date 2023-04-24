from typing import List, Tuple, Dict

class Preprocessor:
    """
    A class for preprocessing datasets.

    Methods:
        convert_to_float(dataset: List[List[str]], column: int) -> None: Convert the values in a specified column to floats.
        convert_to_int(dataset: List[List[str]], column: int) -> Dict[str, int]: Convert the string values in a specified column to integers.
        normalize_dataset(dataset: List[List[float]], minmax: List[Tuple[float, float]]) -> None: Normalize the dataset using min-max normalization.
        dataset_minmax(dataset: List[List[float]]) -> List[Tuple[float, float]]: Calculate the minimum and maximum values for each column.
    """

    @staticmethod
    def convert_to_float(dataset: List[List[str]], column: int) -> None:
        """
        Convert the values in a specified column of a dataset from strings to floats.

        Args:
            dataset (List[List[str]]): The dataset. Each row is a list of values.
            column (int): The index of the column to convert.
        """

        for row in dataset:
            try:
                row[column] = float(row[column].strip())
            except ValueError:
                print(f"Error: Could not convert value '{row[column]}' to float.")

    @staticmethod
    def convert_to_int(dataset: List[List[str]], column: int) -> Dict[str, int]:
        """
        Convert the string values in a specified column of a dataset to integers.

        Args:
            dataset (List[List[str]]): The dataset. Each row is a list of values.
            column (int): The index of the column to convert.

        Returns:
            Dict[str, int]: A dictionary mapping the original string values to their assigned integer values.
        """

        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = {value: i for i, value in enumerate(unique)}
        for row in dataset:
            row[column] = lookup.get(row[column], None)
        return lookup

    @staticmethod
    def normalize_dataset(dataset: List[List[float]], minmax: List[Tuple[float, float]]) -> None:
        """
        Normalize the dataset using the provided minimum and maximum values.

        Args:
            dataset (List[List[float]]): The dataset with numerical values.
            minmax (List[Tuple[float, float]]): A list of (min, max) pairs for each column.
        """

        for row in dataset:
            for i in range(len(row)):
                try:
                    row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
                except ZeroDivisionError:
                    print(f"Warning: Divison by zero encountered in column {i}. Skipping normalization for this value.")
                except TypeError:
                    print(f"Warning: Non-numeric data found in column {i}. Skipping normalization for this value.")


    @staticmethod
    def dataset_minmax(dataset: List[List[float]]) -> List[Tuple[float, float]]:
        """
        Calculate the minimum and maximum values for each column in the dataset.

        Args:
            dataset (List[List[float]]): The dataset with numerical values.

        Returns:
            List[Tuple[float, float]]: A list of (min, max) pairs for each column.
        """

        minmax = []
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append((value_min, value_max))
        return minmax
