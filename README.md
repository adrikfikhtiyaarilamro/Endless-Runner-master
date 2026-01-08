# Endless Runner + Voice Command Integration

Panduan singkat menjalankan game dengan perintah suara (Windows).

## Requirements
- Windows 10/11, .NET 9 SDK
- Python 3.10+ (anjuran), pip
- GTK3 runtime (dibutuhkan oleh TCPServer GTK). Jika belum ada, install GTK3 for Windows (MSYS2 `mingw-w64-x86_64-gtk3` atau installer GTK3 runtime).
- (Opsional) GPU + PyTorch CUDA

## Langkah
1) Aktifkan Python env & install deps
```powershell
cd path\to\Endless-Runner-master
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Jalankan TCP Server (port 5005)
```powershell
cd path\to\Endless-Runner-master
dotnet run --project .\TCPServer\TCPServer.csproj --configuration Debug
```
- Klik **Start Server** di jendela GTK.
- Pastikan port 5005 listening (`netstat -ano | findstr :5005`).

3) Jalankan Voice Inference GUI
```powershell
cd path\to\Endless-Runner-master\Voice
..\.venv\Scripts\activate
python inference_gui.py
```
- Pilih checkpoint di `..\checkpoints\transformer_experiment\bestmodel(white).pth` atau `bestmodel(pink).pth`.
- Pastikan konfigurasi host/port GUI mengarah ke `127.0.0.1:5005`.

4) Jalankan Game Unity
- Buka `Game 3D Endless Runner\Aing Kasep.exe`.

5) Alur Kerja
- GUI Voice: rekam/muat audio → model prediksi komando → kirim teks (`left/right/up/down`) via TCP ke server di port 5005.
- TCP Server: terima teks → log di GUI → gerakkan kotak indikator → simulasi keypress ke jendela game.
- Game Unity: menerima input keyboard dan menggerakkan karakter.

## Tips & Troubleshooting
- **Port 5005 tidak listening**: pastikan tombol **Start Server** diklik dan netstat menunjukkan LISTENING.
- **GTK error / GUI tidak muncul**: pastikan GTK3 runtime ter-install; jalankan ulang `dotnet run` setelah instalasi.
- **Torch/torchaudio gagal install**: gunakan index PyTorch resmi, contoh CUDA 12.4:
  `pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio`
  atau CPU-only: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio`.
- **Game tidak bergerak**: jendela game harus dalam fokus; pastikan TCP log menerima komando.
