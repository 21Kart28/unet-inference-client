import pytest

from mock_server import MockMonaiLabel
from unet_inference_client.backend import MonaiLabelBackend
from unet_inference_client.config import Settings


@pytest.fixture
def mock_server():
    mock = MockMonaiLabel()
    mock.start()
    yield mock
    mock.stop()


@pytest.fixture
def settings(mock_server, tmp_path):
    return Settings(
        server_url=mock_server.url,
        model="deepedit_seg",
        results_dir=str(tmp_path / "results"),
        studies_dir=str(tmp_path / "studies"),
    )


@pytest.fixture
def backend(settings):
    return MonaiLabelBackend(settings)
