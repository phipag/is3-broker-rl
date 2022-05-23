from python_starter import __version__
from python_starter.hello import hello_world


def test_version():
    assert __version__ == "0.1.0"


def test_hello_world(capsys):
    hello_world("Philipp")
    captured = capsys.readouterr()
    assert captured.out == "Hello world! Philipp\n"
