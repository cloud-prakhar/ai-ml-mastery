# Datasets

## Layout

| Folder | Contents | In Git? |
| --- | --- | --- |
| `raw/` | Downloaded, untouched source data | ❌ git-ignored |
| `processed/` | Cleaned and transformed data | ❌ git-ignored |
| `samples/` | Tiny illustrative extracts used inside modules | ✅ committed |

Only `samples/` is committed, and only for files small enough to belong in a repository
(kilobytes, not megabytes). Everything else is downloaded on demand by a script, so the repository
stays clonable on a slow connection.

## 🔐 Rules

- **Never commit personal data.** Not anonymised-in-your-opinion data either.
- **Never commit private or licence-restricted datasets.** Link to the source instead.
- Record for every dataset: source URL, licence, size, and the date you downloaded it.
- Prefer datasets with a clear, permissive licence.
- If a dataset contains personal data, the module using it must say so and explain the handling
  requirements — see [`26-responsible-ai/`](../26-responsible-ai/README.md).

## Adding a dataset

Write a `scripts/download_*.py` that fetches it, verifies a checksum and writes to `raw/`.
Downloading at runtime with verification beats vendoring: it keeps clones fast and makes the
provenance explicit.

---

[🏠 Repository Home](../README.md) · [Data Foundations module →](../03-data-foundations/README.md)
