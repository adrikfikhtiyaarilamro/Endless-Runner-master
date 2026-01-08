# 🎮 Integrasi Voice Command dengan Game Unity (.exe)

## 📋 Overview
Ada **2 cara** untuk integrasi voice command dengan game Unity yang sudah jadi `.exe`:

---

## ✅ **Cara 1: Keyboard Simulation Bridge**


### Alur Kerja:
```
[Microphone] → [Python Voice AI] → [TCP Server] → [Keyboard Simulation] → [Unity Game.exe]
```

### Langkah Setup:

#### 1. Cek Input Settings Game Unity
Buka game `Aing Kasep.exe` dan test keyboard mana yang digunakan:
- **Arrow Keys**: ← → ↑ ↓
- **WASD**: W A S D  
- **Space**: untuk jump

#### 2. Modifikasi TCPServer untuk Keyboard Simulation

Tambahkan Windows API keyboard simulation di `Program.cs`:

```csharp
using System.Runtime.InteropServices;

// Windows API untuk simulate keyboard
[DllImport("user32.dll")]
static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);

const int KEYEVENTF_KEYDOWN = 0x0000;
const int KEYEVENTF_KEYUP = 0x0002;

// Virtual Key Codes
const byte VK_LEFT = 0x25;    // Arrow Left
const byte VK_UP = 0x26;      // Arrow Up
const byte VK_RIGHT = 0x27;   // Arrow Right
const byte VK_DOWN = 0x28;    // Arrow Down
const byte VK_SPACE = 0x20;   // Spacebar
const byte VK_W = 0x57;       // W key
const byte VK_A = 0x41;       // A key
const byte VK_S = 0x53;       // S key
const byte VK_D = 0x44;       // D key

static void SendKeyToGame(string command)
{
    byte key = 0;
    
    // Map voice commands to keyboard
    switch (command.ToLower())
    {
        case "left":
            key = VK_LEFT;  // atau VK_A untuk WASD
            break;
        case "right":
            key = VK_RIGHT; // atau VK_D untuk WASD
            break;
        case "up":
            key = VK_SPACE; // untuk jump (atau VK_UP / VK_W)
            break;
        case "down":
            key = VK_DOWN;  // untuk slide (atau VK_S)
            break;
    }
    
    if (key != 0)
    {
        // Press key
        keybd_event(key, 0, KEYEVENTF_KEYDOWN, UIntPtr.Zero);
        Thread.Sleep(100); // Hold for 100ms
        keybd_event(key, 0, KEYEVENTF_KEYUP, UIntPtr.Zero);
        
        Console.WriteLine($"[GAME] Sent key: {command}");
    }
}
```

Lalu panggil `SendKeyToGame()` di dalam `HandleClient()`:

```csharp
static void HandleClient(TcpClient client)
{
    NetworkStream stream = client.GetStream();
    byte[] buffer = new byte[1024];

    try
    {
        while (true)
        {
            int byteCount = stream.Read(buffer, 0, buffer.Length);
            if (byteCount == 0) break;
            
            string message = Encoding.UTF8.GetString(buffer, 0, byteCount).Trim().ToLower();
            AppendLog("Received: " + message);

            // Update visual feedback
            if (message == "left" || message == "right" || message == "up" || message == "down")
            {
                box.Move(message);
                Application.Invoke(delegate {
                    canvasArea?.QueueDraw();
                });
                
                // ⭐ SEND KEYBOARD TO GAME
                SendKeyToGame(message);
            }
        }
    }
    catch (Exception ex)
    {
        AppendLog("Error: " + ex.Message);
    }
    finally
    {
        client.Close();
        AppendLog("Client disconnected.");
    }
}
```

#### 3. Build TCP Server
```powershell
cd TCPServer
dotnet build
```

#### 4. Testing

**Terminal 1 - Start TCP Server:**
```powershell
cd TCPServer/bin/Debug/net9.0/win-x64
./TCPServer.exe
```
Klik **"Start Server"**

**Terminal 2 - Start Voice Recognition:**
```powershell
cd Voice
C:\Users\Dricky\AppData\Local\Programs\Python\Python311\python.exe inference_gui_transformer.py
```
Klik **"▶ START"**

**Terminal 3 - Start Unity Game:**
```powershell
cd "Game 3D Endless Runner"
./Aing Kasep.exe
```

**Test Flow:**
1. Ucapkan **"left"** → TCP Server simulate keyboard ← → Game bergerak kiri
2. Ucapkan **"right"** → TCP Server simulate keyboard → → Game bergerak kanan
3. Ucapkan **"up"** → TCP Server simulate Space → Game jump
4. Ucapkan **"down"** → TCP Server simulate ↓ → Game slide

---

## 🔧 **Cara 2: Rebuild Unity dengan TCP Receiver** (Advanced)

Jika kamu punya **source code Unity project**, bisa tambahkan TCP receiver langsung di game.

### Script Unity:

Create file: `Assets/Scripts/VoiceCommandReceiver.cs`

```csharp
using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class VoiceCommandReceiver : MonoBehaviour
{
    public PlayerController playerController; // Reference ke player script
    
    private TcpListener tcpListener;
    private Thread tcpThread;
    private bool isRunning = false;
    private string latestCommand = "";
    private object commandLock = new object();
    
    void Start()
    {
        // Start TCP listener in background thread
        tcpThread = new Thread(new ThreadStart(ListenForCommands));
        tcpThread.IsBackground = true;
        isRunning = true;
        tcpThread.Start();
        
        Debug.Log("✓ Voice Command Receiver listening on port 5005");
    }
    
    void Update()
    {
        // Process commands on main Unity thread (not thread-safe otherwise)
        lock(commandLock)
        {
            if (!string.IsNullOrEmpty(latestCommand))
            {
                ProcessCommand(latestCommand);
                latestCommand = "";
            }
        }
    }
    
    void ListenForCommands()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, 5005);
            tcpListener.Start();
            Debug.Log("✓ TCP Listener started");
            
            while (isRunning)
            {
                TcpClient client = tcpListener.AcceptTcpClient();
                Debug.Log("✓ Voice client connected");
                
                Thread clientThread = new Thread(() => HandleClient(client));
                clientThread.IsBackground = true;
                clientThread.Start();
            }
        }
        catch (Exception e)
        {
            Debug.LogError("TCP Error: " + e.Message);
        }
    }
    
    void HandleClient(TcpClient client)
    {
        NetworkStream stream = client.GetStream();
        byte[] buffer = new byte[256];
        
        try
        {
            while (client.Connected && isRunning)
            {
                int bytes = stream.Read(buffer, 0, buffer.Length);
                if (bytes > 0)
                {
                    string command = Encoding.UTF8.GetString(buffer, 0, bytes).Trim().ToLower();
                    
                    lock(commandLock)
                    {
                        latestCommand = command;
                    }
                    
                    Debug.Log($"🎤 Voice command: {command}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Client error: " + e.Message);
        }
        finally
        {
            client.Close();
            Debug.Log("✗ Voice client disconnected");
        }
    }
    
    void ProcessCommand(string cmd)
    {
        // Map voice commands to player actions
        switch(cmd)
        {
            case "up":
                if (playerController != null)
                    playerController.Jump();
                Debug.Log("→ Player JUMP");
                break;
                
            case "down":
                if (playerController != null)
                    playerController.Slide();
                Debug.Log("→ Player SLIDE");
                break;
                
            case "left":
                if (playerController != null)
                    playerController.MoveLeft();
                Debug.Log("→ Player MOVE LEFT");
                break;
                
            case "right":
                if (playerController != null)
                    playerController.MoveRight();
                Debug.Log("→ Player MOVE RIGHT");
                break;
                
            default:
                Debug.LogWarning($"Unknown command: {cmd}");
                break;
        }
    }
    
    void OnApplicationQuit()
    {
        isRunning = false;
        
        if (tcpListener != null)
        {
            tcpListener.Stop();
            Debug.Log("✓ TCP Listener stopped");
        }
        
        if (tcpThread != null && tcpThread.IsAlive)
        {
            tcpThread.Join(1000); // Wait max 1 second
        }
    }
}
```

### Setup di Unity Editor:

1. Attach script `VoiceCommandReceiver.cs` ke **Main Camera** atau **GameManager** GameObject
2. Drag player GameObject ke field **PlayerController** di Inspector
3. Build game → File > Build Settings > Build
4. Test dengan Python voice recognition

---

## 🎯 Recommended Keyboard Mapping

Untuk Endless Runner game, mapping yang umum:

| Voice Command | Unity Input | Action |
|--------------|-------------|---------|
| `"up"` | **Space** atau **↑** | Jump / Jump over obstacle |
| `"down"` | **↓** atau **S** | Slide / Duck under obstacle |
| `"left"` | **←** atau **A** | Move to left lane |
| `"right"` | **→** atau **D** | Move to right lane |

---

## 📊 Testing Checklist

- [ ] TCP Server berjalan di port 5005
- [ ] Voice recognition mendeteksi command dengan benar
- [ ] TCP Server menerima command dari Python
- [ ] Keyboard simulation berfungsi (coba manual di Notepad dulu)
- [ ] Unity game merespon keyboard input
- [ ] Latency acceptable (~100-200ms)
- [ ] Game tidak crash saat TCP disconnect

---

## 🐛 Troubleshooting

### Problem: Game tidak merespon voice command
**Solution:**
1. Test keyboard manual di game (press arrow keys)
2. Check apakah game window dalam focus (klik game window)
3. Verify keyboard mapping di Unity Input Settings
4. Check logs di TCP Server untuk konfirmasi keyboard sent

### Problem: Keyboard simulation tidak kerja
**Solution:**
1. Run TCP Server **as Administrator** (Windows security)
2. Pastikan game window aktif (tidak minimized)
3. Test dengan Notepad terlebih dahulu
4. Gunakan Virtual Key Code yang benar sesuai keyboard layout

### Problem: Lag antara voice dan action
**Solution:**
1. Reduce `Thread.Sleep()` duration di `SendKeyToGame()`
2. Optimize voice recognition (lower RMS threshold)
3. Check CPU usage tidak overload

---

## 🚀 Quick Start Commands

```powershell
# Terminal 1: TCP Server
cd TCPServer/bin/Debug/net9.0/win-x64
./TCPServer.exe

# Terminal 2: Voice Recognition
cd Voice
C:\Users\Dricky\AppData\Local\Programs\Python\Python311\python.exe inference_gui_transformer.py

# Terminal 3: Unity Game
cd "Game 3D Endless Runner"
./Aing Kasep.exe
```

**Test sequence:**
1. Klik "Start Server" di TCP Server GUI
2. Klik "▶ START" di Voice Recognition GUI
3. Launch game Unity
4. Ucapkan: "left", "right", "up", "down"

---

## ✅ Status

- [x] Python Voice Recognition - READY
- [x] TCP Client implementation - READY
- [x] TCP Server - READY
- [ ] Keyboard simulation - NEEDS IMPLEMENTATION
- [ ] Unity game integration - PENDING TEST

**Next Step:** Implement keyboard simulation di `Program.cs` atau rebuild Unity dengan TCP receiver script.
