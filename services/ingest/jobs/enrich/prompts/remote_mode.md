Decide work mode from the text.

Return **ONLY** JSON:

{"remote_mode":"remote|hybrid|onsite"}

Rules:
- "work from anywhere", "remote-first", timezone-only ⇒ remote.
- Specific office days or mixed wording ⇒ hybrid.
- Otherwise ⇒ onsite.
