from service import fetch
from storage import Storage


def test_reads_existing_key():
    assert fetch(Storage({'a': 1}), 'a') == 1
