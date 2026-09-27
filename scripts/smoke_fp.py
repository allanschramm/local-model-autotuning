import autoresearch_core.fingerprint as fp

print("dir:", dir(fp))
print("has dump:", hasattr(fp, "dump"))
print("has load:", hasattr(fp, "load"))
print("has mismatch_reason:", hasattr(fp, "mismatch_reason"))
print("has FingerprintError:", hasattr(fp, "FingerprintError"))
