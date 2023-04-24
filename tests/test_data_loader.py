import pytest
from knn.data_loader import DataLoader

def test_load_data_with_header(tmpdir):
    # Create a temporary CSV file
    file = tmpdir.join("test.csv")
    file.write("col1,col2\n1,2\n3,4\n")

    loader = DataLoader(str(file), skip_header=False)
    data = loader.load_data()

    assert isinstance(data, list)  # Check if data is a list
    assert len(data) == 3  # Ensure the correct number of rows is loaded
    assert data == [["col1", "col2"], ["1", "2"], ["3", "4"]]  # Verify the content

def test_load_data_without_header(tmpdir):
    # Create a temporary CSV file
    file = tmpdir.join("test.csv")
    file.write("col1,col2\n1,2\n3,4\n")

    loader = DataLoader(str(file), skip_header=True)
    data = loader.load_data()

    assert isinstance(data, list)  # Check if data is a list
    assert len(data) == 2  # Ensure the correct number of rows is loaded
    assert data == [["1", "2"], ["3", "4"]]  # Verify the content

def test_file_not_found():
    loader = DataLoader('non_existent_file.csv')
    with pytest.raises(FileNotFoundError):
        loader.load_data()


