"""Security regression: reject executable pickle opcodes, trailing data, malformed
JSON, non-list cases and decompression bombs before any decoded value is used.
The importer accepts one literal JSON string, never executable pickle objects.
"""

import base64
import json
import pickle
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from import_lcb import MAX_DECODED_BYTES, decode_private_cases


def wrapped(value):
    return base64.b64encode(zlib.compress(pickle.dumps(value, protocol=4))).decode()


with tempfile.TemporaryDirectory(prefix="lcb-decoder-check-") as directory:
    marker = Path(directory) / "executed.txt"

    class Executable:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    valid = [{"input": "1", "output": "2\n", "testtype": "stdin"}]
    assert decode_private_cases(wrapped(json.dumps(valid))) == valid
    raw = pickle.dumps(json.dumps(valid), protocol=4) + b"trailing"
    invalid = [
        wrapped(Executable()),
        wrapped("not json"),
        wrapped(json.dumps({"input": "1"})),
        wrapped(json.dumps([])),
        wrapped(json.dumps([{"input": "1", "output": 2, "testtype": "stdin"}])),
        wrapped(json.dumps([{"input": "1", "output": "2", "testtype": "functional"}])),
        base64.b64encode(zlib.compress(raw)).decode(),
        base64.b64encode(zlib.compress(b"x" * (MAX_DECODED_BYTES + 1))).decode(),
        "invalid base64!",
    ]
    for payload in invalid:
        try:
            decode_private_cases(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe or malformed private cases accepted")
    assert not marker.exists(), "Decoder executed an external pickle callable"
print("LCB importer security passed: 1 valid, 9 rejected, no code execution")
