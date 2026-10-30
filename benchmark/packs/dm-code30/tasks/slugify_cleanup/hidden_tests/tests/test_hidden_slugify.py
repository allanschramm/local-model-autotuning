from text_utils import slugify


def test_punctuation_and_repeated_spaces():
    assert slugify("Python & AI Agents!") == "python-ai-agents"
    assert slugify("many     spaces") == "many-spaces"


def test_strips_generated_separators():
    assert slugify("--Hello__World--") == "hello-world"
    assert slugify("!!!") == ""
