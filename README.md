# Confluence → Qdrant fetch-all

Mengambil **semua halaman** dari satu atau lebih space Confluence dan memasukkannya ke Qdrant (hybrid: dense + sparse BM25).
Dijalankan sekali per eksekusi, tanpa scheduler. Untuk sinkronisasi harian lihat
`../confluence-qdrant-ingest`.

## Alur

1. `GET /rest/api/content?spaceKey=<KEY>&type=page` per space, dipaginasi lewat `_links.next`.
2. Tiap halaman: `GET /api/v2/pages/{id}?body-format=export_view`.
3. Body kosong dilewati.
4. Convert HTML → Markdown (`CONVERT_API_URL`) → chunk (`CHUNK_API_URL`, 6000/200).
5. Embedding dense per batch `EMBEDDING_BATCH_SIZE` chunk (satu request per batch).
6. Hapus point lama dengan `pageId` yang sama, lalu upsert point baru.

Payload per point: `text, chunk_index, token_count, word_count, pageId, title, spaceKey, url, createdAt, lastModified`.
Sparse BM25 dihitung Qdrant (inference object `qdrant/bm25`, `options.avg_len` dari chunk API).

## Perilaku penting

- **Re-run tidak menduplikasi.** Point id berupa UUID acak, jadi sebelum upsert point lama
  untuk `pageId` itu dihapus (delete by filter, tanpa query). Hapus dilakukan setelah convert,
  chunk, dan embed berhasil, jadi kalau salah satunya gagal, vektor lama tetap utuh.
- Halaman yang gagal dicatat, run lanjut ke halaman berikutnya, dan exit code 1 di akhir.
- Mendukung banyak space (`CONFLUENCE_SPACE_KEYS=II,SP`).
- Collection dibuat lewat flag `--create-collection`, hanya bila belum ada.
- Semua URL, parameter, dan credential dibaca dari `.env`. Tidak ada nilai infra yang di-hardcode di kode.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # lalu isi credential
```

Wajib diisi di `.env`: `CONFLUENCE_EMAIL`, `CONFLUENCE_API_TOKEN`, `EMBEDDING_API_KEY`, `QDRANT_API_KEY`.
Kalau sudah punya `.env` di `../confluence-qdrant-ingest`, nilainya bisa disalin.
Daftar lengkap variabel ada di [.env.example](.env.example). `.env` sudah di-gitignore.

## Menjalankan

```bash
.venv/bin/python -m fetchall --dry-run                  # hanya daftar halaman, tidak menulis apa pun
.venv/bin/python -m fetchall --create-collection --limit 3   # uji 3 halaman (buat collection jika belum ada)
.venv/bin/python -m fetchall                            # semua halaman
```

`--create-collection` membuat collection (`dense` 1536 Cosine + `sparse` idf) hanya bila belum ada.
Ukuran dense diatur `EMBEDDING_DIMENSIONS` dan harus cocok dengan `EMBEDDING_MODEL`.

## Tes

```bash
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest
```

Tes memakai mock HTTP (tanpa jaringan).
