from text_utils import slugify


def test_basic_words():
    assert slugify("Hello World") == "hello-world"


def test_trims_edges():
    assert slugify("  Already Slug  ") == "already-slug"
