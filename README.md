# Chatbot-bhs.indonesia
Project ini adalah implementasi sistem chatbot/LLM berbasis arsitektur TinyGPT yang diotimalkan untuk percakapan dalam bahasa Indonesia.

**Informasi:**
Model (`best.pt`) memiliki jumlah parameter sebesar **33 Juta**, sangat cepat dan efisien untuk dijalankan walau hanya menggunakan CPU.

## Cara Menjalankan Chatbot

Untuk mulai chatting dengan model bot (`best.pt`), Anda bisa menjalankan `chat.py` di terminal (Command Prompt/PowerShell).

Pastikan *dependencies* Anda (utamanya `torch` dan `sentencepiece`) sudah terpasang.

Jika belum menginstall *dependencies* :
```bash
pip install torch sentencepiece
```

### Chat Dengan Model AI
Jalankan command ini untuk memulai sesi percakapan secara interaktif:

```bash
python chat.py --checkpoint best.pt --tokenizer spm.model --config config.json --interactive
```

*Jika Anda menggunakan Command Prompt di Windows, Anda bisa menggunakan tanda `^` untuk memecah baris command agar mudah dibaca:*
```cmd
python chat.py ^
  --checkpoint best.pt ^
  --tokenizer spm.model ^
  --config config.json ^
  --interactive
```

### Argumen Penting di `chat.py`

- `--checkpoint` : Menentukan path/lokasi dari model (`best.pt`).
- `--tokenizer` : Path tokenizer sentencepiece (`spm.model`).
- `--config` : Mengarah ke konfigurasi struktur model (`config.json`).
- `--interactive`: Flag untuk masuk mode percakapan berlanjut. Jika tidak ditulis, diperlukan flag `--prompt "Pertanyaan anda"`.
- `--temperature`: Nilai (default `0.8`) untuk mengukur tingkat kreativitas bot. Semakin rendah, semakin kaku; semakin tinggi, semakin bervariasi jenis kata yang digunakan.
- `--cpu`: Paksa menggunakan CPU untuk kalkulasi, biarpun Anda mempunyai laptop/PC dengan GPU NVIDIA.

### Command Khusus di dalam Chat

Saat teks `Anda :` muncul di layar, selain mengetik kalimat tanya/pernyataan biasa, Anda bisa mengetik command khusus bawaan dari CLI chatbot ini:
- `/reset` : Menghapus semua riwayat percakapan (membuat chatbot lupa tentang apa yang dibincangkan sebelumnya).
- `/history` : Menampilkan ringkasan obrolan barusan.
- `/exit` atau `/quit` : Menutup aplikasi dan kembali ke layar terminal.
