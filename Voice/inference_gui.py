import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import math
import socket
import os
import csv
import time
from collections import deque

# === TCP Settings ===
TCP_HOST = '127.0.0.1'
TCP_PORT = 5005

socketClient = None

# === Model Settings ===
SAMPLE_RATE = 16000
DURATION = 1.0 
N_MFCC = 40  
N_MELS = 40
NUM_CLASSES = 4
LABELS = ['down', 'left', 'right', 'up']

# Decision thresholds to reduce wrong feedback
CONF_THRESH = 0.6           # minimum confidence to consider a command
HIGH_CONF_THRESH = 0.8      # auto-commit if above this
TOP2_MARGIN_MIN = 0.15      # require top1 - top2 margin to be at least this
AGREE_WINDOW = 3            # consecutive predictions window
AGREE_MIN = 2               # require at least this many consecutive agreements

# MFCC parameters for 16kHz
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../checkpoints/transformer/best_model_40.pth')

# === BiLSTM Settings ===
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
USE_BILSTM = False  

# === Model Definition ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pe = self.pe[:, :T, :].to(dtype=x.dtype, device=x.device)
        return self.dropout(x + pe)


class MFCC_Transformer(nn.Module):
    def __init__(
        self,
        n_mfcc: int = 40,
        num_classes: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_len: int = 4096,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.n_mfcc = int(n_mfcc)

        self.input_proj = nn.Linear(self.n_mfcc, d_model, bias=True)
        self.posenc = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B,T,{self.n_mfcc}] or [B,{self.n_mfcc},T], got {tuple(x.shape)}")
        B, A, C = x.shape
        if C == self.n_mfcc:
            pass
        elif A == self.n_mfcc:
            x = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Input last dim must be n_mfcc={self.n_mfcc}; got {tuple(x.shape)}")

        h = self.input_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        feat = h.mean(dim=1)
        logits = self.head(feat)
        return logits

class BiLSTM_Transformer(nn.Module):
    def __init__(self,
                 n_mfcc: int,
                 num_classes: int,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 bidirectional: bool = True,
                 lstm_dropout: float = 0.2,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.3,
                 max_len: int = 4096):
        super().__init__()
        self.n_mfcc = int(n_mfcc)
        self.bidir = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=self.n_mfcc,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            bidirectional=self.bidir
        )
        lstm_out = lstm_hidden * (2 if self.bidir else 1)
        self.proj = nn.Linear(lstm_out, d_model)
        self.posenc  = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        enc_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,n_mfcc] / [B,n_mfcc,T]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.shape[-1] == self.n_mfcc:
            seq = x                                
        elif x.shape[1] == self.n_mfcc:
            seq = x.transpose(1, 2).contiguous()    
        else:
            raise ValueError(f"Input last dim must be n_mfcc={self.n_mfcc}; got {tuple(x.shape)}")

        lstm_out, _ = self.lstm(seq)                
        h = self.proj(lstm_out)                     
        h = self.posenc(h)
        h = self.encoder(h)
        feat = h.mean(dim=1)
        return self.head(feat)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if USE_BILSTM:
    model = BiLSTM_Transformer(
        n_mfcc=N_MFCC, 
        num_classes=NUM_CLASSES,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        lstm_dropout=LSTM_DROPOUT
    )
    print("Initializing BiLSTM_Transformer...")
else:
    model = MFCC_Transformer(n_mfcc=N_MFCC, num_classes=NUM_CLASSES)
    print("Initializing MFCC_Transformer...")

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print(f"  Please check if the model file exists at: {MODEL_PATH}")

model.to(device)
model.eval()

# === MFCC Transform ===
# Move to device to match training pipeline
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "center": True,
        "f_min": 0.0,
        "f_max": SAMPLE_RATE / 2
    }
).to(device)

def send_tcp_command(command):
    """Send command via TCP and wait for ACK. Returns (send_duration, ack_latency, game_status)."""
    try:
        if socketClient:
            full_msg = f"{command.upper()}\n"
            t0_send = time.perf_counter()
            socketClient.sendall(full_msg.encode('utf-8'))
            t1_send = time.perf_counter()
            send_duration = t1_send - t0_send

            ack_latency = None
            game_status = "unknown"  # Track game status: success/blocked/unknown
            try:
                # Temporarily set a timeout for ACK reception
                socketClient.settimeout(1.0)
                buf = b""
                while True:
                    chunk = socketClient.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                    # Stop after first line or if buffer starts with ACK
                    if b"\n" in buf or buf.startswith(b"ACK"):
                        break
                if buf:
                    line = buf.split(b"\n")[0].decode("utf-8", errors="ignore").strip()
                    if line.startswith("ACK"):
                        t_ack = time.perf_counter()
                        # Measure ACK latency relative to end of send to avoid double counting
                        ack_latency = t_ack - t1_send
                        # Parse ACK format: "ACK|command|timestamp|game_status"
                        parts = line.split("|")
                        if len(parts) >= 4:
                            game_status = parts[3]  # Extract game_status
                        print(f"[TCP] Received: {line} (ACK latency {ack_latency*1000:.2f} ms, Status: {game_status})")
            except Exception as e:
                print(f"[TCP] ACK wait error: {e}")
            finally:
                try:
                    socketClient.settimeout(None)
                except Exception:
                    pass

            print(f"[TCP] Sent: {command}")
            return send_duration, ack_latency, game_status
    except Exception as e:
        print(f"[TCP] Error: {e}")
    return 0.0, None, "error"

import soundfile as sf 
SAVE_DIR = os.path.join(BASE_DIR, "captured_samples")
os.makedirs(SAVE_DIR, exist_ok=True)
save_samples_var = None

def save_audio_sample(audio_np, label, prob):
    """Save the audio sample to disk"""
    if not (save_samples_var and save_samples_var.get()):
        return
        
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SAVE_DIR, f"{timestamp}_{label}_{prob:.2f}.wav")
        
        # Use soundfile directly to avoid torchaudio backend issues
        # audio_np is (1, T), soundfile expects (T,) or (T, 1)
        data_to_save = audio_np.flatten()
        sf.write(filename, data_to_save, SAMPLE_RATE)
        
        print(f"[DATA] Saved {filename}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[DATA] Failed to save sample: {e}")

# === Logging Setup ===
LOG_FILE = "inference_log_detailed.csv"
experiment_id = 0

def init_csv_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Headers matching user request + timestamps/confidence + game_status
            headers = [
                "ID Percobaan", 
                "Timestamp", 
                "Perintah Suara (Label - Manual)", 
                "Prediksi Model", 
                "Status (Benar/Salah - Manual)", 
                "Confidence", 
                "Inference Time (ms)", 
                "Transport Latency (ms)", 
                "Server ACK Latency (ms)",
                "Total Response Time (ms)",
                "Game Status"  # NEW: Track game movement success/blocked
            ]
            writer.writerow(headers)
            print(f"[LOG] Created {LOG_FILE}")

init_csv_log()

def log_inference(prediction, confidence, inf_time, transport_time, ack_latency=None, game_status="unknown"):
    global experiment_id
    experiment_id += 1
    total_time = inf_time + transport_time + (ack_latency if ack_latency is not None else 0.0)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                experiment_id, 
                timestamp, 
                "",             
                prediction, 
                "",             
                f"{confidence:.2f}", 
                f"{inf_time*1000:.2f}", 
                f"{transport_time*1000:.2f}", 
                f"{(ack_latency*1000 if ack_latency is not None else 0.0):.2f}",
                f"{total_time*1000:.2f}",
                game_status  # NEW: Log game status
            ])
    except Exception as e:
        print(f"[LOG] Error writing to CSV: {e}")

is_listening = False
last_prediction_time = 0  # Track last prediction to prevent duplicates
COOLDOWN_SECONDS = 0.8    # Cooldown period after successful prediction
pred_history = deque(maxlen=AGREE_WINDOW)

def continuous_listen():
    """Continuously listen and predict"""
    global is_listening, last_prediction_time
    window_size = int(0.05 * SAMPLE_RATE)

    while is_listening:
        # Check cooldown - skip if too soon after last prediction
        time_since_last = time.time() - last_prediction_time
        if time_since_last < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - time_since_last
            result_var.set(f"⏳ Cooldown {remaining:.1f}s...")
            time.sleep(0.1)
            continue
        
        frames = []


        def callback(indata, frames_count, time, status):
            amplified = indata
            rms = np.sqrt(np.mean(amplified**2))
            volume_level.set(min(rms * 500, 100))
            frames.append(amplified.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=window_size, callback=callback):
            sd.sleep(int(DURATION * 1000))

        # Process audio
        audio = np.concatenate(frames, axis=0).T
        audio = np.clip(audio, -1.0, 1.0)
        
        if audio.shape[1] < SAMPLE_RATE:
            pad_length = SAMPLE_RATE - audio.shape[1]
            audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
        elif audio.shape[1] > SAMPLE_RATE:
            audio = audio[:, :SAMPLE_RATE]
        
        global_rms = np.sqrt(np.mean(audio**2))

        print(f"[DEBUG] RMS: {global_rms:.4f}, Audio samples: {audio.shape[1]}")

        # Lower threshold for better mic sensitivity
        if global_rms < 0.002: 
            result_var.set(f"No sound detected (RMS: {global_rms:.4f})")
            continue

        waveform = torch.tensor(audio, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize to [-1, 1] range for consistent MFCC extraction
        max_val = waveform.abs().max()
        if max_val > 1e-6:
            waveform = waveform / max_val

        print(f"[DEBUG] Waveform shape: {waveform.shape}")
        print(f"[DEBUG] Waveform min/max: {waveform.min():.4f} / {waveform.max():.4f}")
        
        features = mfcc_transform(waveform) 
        features = features.transpose(1, 2).contiguous()  
        features = features.to(device)

        print(f"[DEBUG] Features shape: {features.shape}")
        print(f"[DEBUG] Features min/max/mean: {features.min():.2f} / {features.max():.2f} / {features.mean():.2f}")



        t0_inf = time.perf_counter()
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            max_prob, pred = torch.max(probs, dim=1)

            # Show all class probabilities
            prob_str = " | ".join([f"{LABELS[i]}: {probs[0][i].item():.3f}" for i in range(len(LABELS))])
            print(f"[PROBS] {prob_str}")
            print(f"[PRED] {LABELS[pred.item()].upper()} with confidence {max_prob.item():.4f}")

            t1_inf = time.perf_counter()
            inference_dur = t1_inf - t0_inf
            transport_dur = 0.0

            # Robust decision: require sufficient confidence, top-2 margin,
            # and consecutive agreement, else mark as Uncertain
            probs_sorted, indices = torch.sort(probs[0], descending=True)
            top1_label = LABELS[indices[0].item()]
            top2_label = LABELS[indices[1].item()]
            top1_prob = probs_sorted[0].item()
            top2_prob = probs_sorted[1].item()
            margin = top1_prob - top2_prob

            should_commit = False
            agree_count = 1
            # High confidence -> commit immediately regardless of margin
            if top1_prob >= HIGH_CONF_THRESH:
                should_commit = True
            else:
                # Accumulate history and require agreement within window
                pred_history.append((top1_label, top1_prob))
                agree_count = sum(1 for lbl, _p in pred_history if lbl == top1_label)
                if margin >= TOP2_MARGIN_MIN and top1_prob >= CONF_THRESH and agree_count >= AGREE_MIN:
                    should_commit = True

            if not should_commit:
                result_var.set(f"❓ Uncertain: {top1_label}({top1_prob:.2f}) vs {top2_label}({top2_prob:.2f})")
            else:
                prediction = top1_label
                result_var.set(f"✅ {prediction.upper()} ({top1_prob:.2f})")
                
                send_dur, ack_latency, game_status = send_tcp_command(prediction)
                transport_dur = send_dur
                
                # Update last prediction time to start cooldown
                last_prediction_time = time.time()
                print(f"[COOLDOWN] Started {COOLDOWN_SECONDS}s cooldown after '{prediction}' (prob={top1_prob:.2f}, margin={margin:.2f}, agree={agree_count})")
                
            # Log Data (use top1_prob)
            current_pred = (prediction if should_commit else "Uncertain")
            # Pass ACK latency and game status if available
            try:
                game_status_log = game_status if should_commit else "uncertain"
                log_inference(current_pred, top1_prob, inference_dur, transport_dur, ack_latency if should_commit else None, game_status_log)
            except NameError:
                # Fallback if game_status isn't defined (uncertain path)
                log_inference(current_pred, top1_prob, inference_dur, transport_dur, None)

            # Save data if enabled
            save_audio_sample(audio, LABELS[pred.item()], max_prob.item())



def start_listening():
    """Start listening mode"""
    global is_listening, socketClient

    if not is_listening:
        try:
            socketClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socketClient.connect((TCP_HOST, TCP_PORT))
            print("[TCP] Connected")
        except Exception as e:
            print(f"[TCP] Connection failed: {e}")
            print("[TCP] Running without TCP server (voice recognition still works)")
            socketClient = None

        is_listening = True
        result_var.set("🎧 Listening...")
        threading.Thread(target=continuous_listen, daemon=True).start()


def stop_listening():
    """Stop listening mode"""
    global is_listening, socketClient, last_prediction_time
    is_listening = False
    last_prediction_time = 0  # Reset cooldown

    if socketClient:
        try:
            socketClient.close()
            print("[TCP] Closed")
        except Exception as e:
            pass

    socketClient = None
    result_var.set("⏹ Stopped Listening.")


# GUI
root = tk.Tk()
root.title("Voice Command Transformer")
root.configure(bg="#2e2e2e")

volume_level = tk.DoubleVar()
result_var = tk.StringVar()
result_var.set("Press Start to begin")

# Style configuration
style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background="#2e2e2e")
style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Arial", 12))
style.configure("TButton", background="#444", foreground="white", font=("Arial", 10))
style.map("TButton", background=[("active", "#666")])
style.configure("TProgressbar", background="#4caf50")

frame = ttk.Frame(root, padding=20)
frame.grid()

title = ttk.Label(frame, text="Voice Command Classifier (Transformer)", font=("Arial", 16, "bold"))
title.grid(column=0, row=0, columnspan=2, pady=10)

model_info = ttk.Label(frame, text=f"Model: MFCC Transformer | SR: {SAMPLE_RATE}Hz | MFCC: {N_MFCC}", 
                       font=("Arial", 9))
model_info.grid(column=0, row=1, columnspan=2, pady=5)

start_btn = ttk.Button(frame, text="▶ Start Listening", command=start_listening)
start_btn.grid(column=0, row=2, pady=10, padx=5)

stop_btn = ttk.Button(frame, text="⏹ Stop Listening", command=stop_listening)
stop_btn.grid(column=1, row=2, pady=10, padx=5)

volume_label = ttk.Label(frame, text="Volume Level:", font=("Arial", 10))
volume_label.grid(column=0, row=3, columnspan=2, pady=(10, 0))

progress = ttk.Progressbar(frame, orient='horizontal', length=300,
                           mode='determinate', maximum=100,
                           variable=volume_level)
progress.grid(column=0, row=4, columnspan=2, pady=5)

result_label = ttk.Label(frame, textvariable=result_var, font=("Arial", 14, "bold"))
result_label.grid(column=0, row=5, columnspan=2, pady=15)

classes_label = ttk.Label(frame, text=f"Classes: {', '.join(LABELS)}", font=("Arial", 9))
classes_label.grid(column=0, row=6, columnspan=2, pady=5)

# Save Checkbox
save_samples_var = tk.BooleanVar(value=False)
save_check = ttk.Checkbutton(frame, text="Save Recorded Samples (for Data Collection)", 
                             variable=save_samples_var, onvalue=True, offvalue=False)
save_check.grid(column=0, row=7, columnspan=2, pady=10)

# Refresh UI loop
def update_gui():
    root.after(50, update_gui)

update_gui()

print(f"\n{'='*60}")
print("Voice Command Transformer - Inference GUI")
print(f"{'='*60}")
print(f"Model: {MODEL_PATH}")
print(f"Device: {device}")
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"MFCC: {N_MFCC} coefficients")
print(f"Classes: {LABELS}")
print(f"{'='*60}\n")

root.mainloop()
