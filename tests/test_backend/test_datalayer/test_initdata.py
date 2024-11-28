import pytest
from unittest import mock
import pandas as pd
from src.backend.datalayer.initdata import (
    DataLayer,
    DataLayerException,
    FileUnreadableError,
)


@pytest.fixture
def data_layer():
    """Fixture pour créer une instance de DataLayer avant chaque test."""
    return DataLayer()


def test_load_csv_file_not_found(data_layer):
    """Test si FileNotFoundError est levée pour un fichier CSV inexistant."""
    with mock.patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            data_layer.load_csv("non_existing_file.csv")


def test_load_csv_file_unreadable(data_layer):
    """Test si FileUnreadableError est levée pour un fichier CSV non lisible."""
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=False
    ):
        with pytest.raises(FileUnreadableError):
            data_layer.load_csv("unreadable_file.csv")


def test_load_csv_empty_file(data_layer):
    """Test si FileUnreadableError est levée pour un fichier CSV vide."""
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=True
    ), mock.patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError):
        with pytest.raises(FileUnreadableError):
            data_layer.load_csv("empty_file.csv")


def test_load_csv_generic_exception(data_layer):
    """Test si une exception générique est encapsulée dans DataLayerException."""
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=True
    ), mock.patch("pandas.read_csv", side_effect=ValueError("Erreur inconnue")):
        with pytest.raises(DataLayerException) as excinfo:
            data_layer.load_csv("generic_error_file.csv")
        assert "Erreur lors du chargement du fichier CSV generic_error_file.csv" in str(
            excinfo.value
        )
        assert "Erreur inconnue" in str(excinfo.value)


def test_load_pickle_file_not_found(data_layer):
    """Test si FileNotFoundError est levée pour un fichier pickle inexistant."""
    with mock.patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            data_layer.load_pickle("non_existing_file.pkl")


def test_load_pickle_file_unreadable(data_layer):
    """Test si FileUnreadableError est levée pour un fichier pickle non lisible."""
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=False
    ):
        with pytest.raises(FileUnreadableError):
            data_layer.load_pickle("unreadable_file.pkl")


def test_load_pickle_empty_file(data_layer):
    """Test si FileUnreadableError est levée pour un fichier pickle vide ou corrompu."""
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=True
    ), mock.patch(
        "pandas.read_pickle", side_effect=pd.errors.EmptyDataError
    ), mock.patch(
        "builtins.open", mock.mock_open(read_data=b"")  # Simule un fichier vide
    ):
        with pytest.raises(FileUnreadableError):
            data_layer.load_pickle("empty_file.pkl")


def test_load_pickle_generic_exception(data_layer):
    """Test si une exception générique est encapsulée dans DataLayerException."""
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=True
    ), mock.patch(
        "pandas.read_pickle", side_effect=ValueError("Erreur inconnue")
    ), mock.patch(
        "builtins.open",
        mock.mock_open(read_data=b"mocked data"),  # Simule un fichier existant
    ):
        with pytest.raises(DataLayerException) as excinfo:
            data_layer.load_pickle("generic_error_file.pkl")
        assert (
            "Erreur lors du chargement du fichier pickle generic_error_file.pkl"
            in str(excinfo.value)
        )
        assert "Erreur inconnue" in str(excinfo.value)


def test_load_data_success(data_layer):
    """Test si la méthode load_data charge correctement les fichiers mockés."""
    # Mocker toutes les lectures de fichiers CSV et pickle
    mock_csv_data = pd.DataFrame({"column": [1, 2, 3]})
    mock_pickle_data = {"key": "value"}

    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.access", return_value=True
    ), mock.patch("pandas.read_csv", return_value=mock_csv_data), mock.patch(
        "pandas.read_pickle", return_value=mock_pickle_data
    ), mock.patch(
        "builtins.open", mock.mock_open(read_data=b"mocked data")
    ):
        data_layer.load_data()

        assert data_layer.get_interactions_test().equals(mock_csv_data)
        assert data_layer.get_interactions_train().equals(mock_csv_data)
        assert data_layer.get_interactions_validation().equals(mock_csv_data)
        assert data_layer.get_pp_recipes().equals(mock_csv_data)
        assert data_layer.get_pp_users().equals(mock_csv_data)
        assert data_layer.get_raw_interactions().equals(mock_csv_data)
        assert data_layer.get_raw_recipes().equals(mock_csv_data)
        assert data_layer.get_ingr_map() == mock_pickle_data
