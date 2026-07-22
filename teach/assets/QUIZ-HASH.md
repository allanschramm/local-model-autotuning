# Quiz hash tooling (maintainers only)

Student HTML stores `data-answer-hash`, never the letter.

Payload: `sha256("teach-lmai-v1|{lessonId}|{A|B|C|D}")`

```powershell
.\venv\Scripts\python.exe -c "import hashlib; s='teach-lmai-v1'; lid='s1d1-q1'; c='B'; print(hashlib.sha256(f'{s}|{lid}|{c}'.encode()).hexdigest())"
```

IDs atuais Semana 1: `s1d1-q1`…`s1d1-q3`, `s1d2-q1`…`s1d2-q2`, `s1d3-q1`…`s1d3-q2`, `s1d4-q1`…`s1d4-q2`

Do **not** commit a plaintext answer key file.
