import logging

from unet_inference_client import logs


def test_configure_writes_to_a_file_in_the_given_folder(tmp_path):
    path = logs.configure(tmp_path / "logs")
    assert path == tmp_path / "logs" / logs.LOG_NAME
    logging.getLogger("unet_inference_client.test").info("hello from the test")
    for handler in logs.logger.handlers:
        handler.flush()
    text = path.read_text(encoding="utf-8")
    assert "INFO    unet_inference_client.test: hello from the test" in text


def test_configure_twice_adds_one_handler(tmp_path):
    logs.configure(tmp_path / "logs")
    before = len(logs.logger.handlers)
    logs.configure(tmp_path / "logs")
    assert len(logs.logger.handlers) == before
