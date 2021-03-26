"""Performs general tests."""
from mcsce.libs import libio


def test_libio_hello():
    """Test libio hello."""
    assert libio.hello()
