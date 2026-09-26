# Hailo-10H NPU — how it's used & how to fix it

The Hailo-10H is Stella's **on-device AI accelerator**. Its value here is **free, private,
offline** inference — she keeps working with no internet.

## What runs on it
- **Offline LLM brain:** `qwen2.5-instruct:1.5b` (Hailo-compiled `.hef`), served by the
  `hailo-ollama` service on `http://localhost:8000`. It's the **last fallback** in the AI
  chain, so if Groq (cloud) is down/rate-limited, Stella still thinks — locally.
  - `llama3.2:3b` is also installed but **too slow** on this NPU (~19s+, times out) — do not
    use it as the live model. `qwen2.5-instruct:1.5b` is the responsive one (warm ≈ 2–5 s;
    first/cold load ≈ 40 s, absorbed by a background warmup at startup).
- Vision/object detection is **not** on the Hailo (would contend for the single NPU context
  with the LLM). Vision uses cloud Moondream; face recognition uses CPU dlib.

## The API gotcha (important)
`hailo-ollama` 5.1.1 serves **only** the OpenAI-compatible endpoint:
`POST /v1/chat/completions` (response = `choices[0].message.content`).
The ollama-native `/api/chat` and `/api/generate` return **empty / errors** on this build.
The app (`modules/ai/ai_engine.py`) was fixed to use `/v1/chat/completions` everywhere.
`/api/tags` still works (used to list local models).

## The kernel-upgrade gotcha (this WILL recur)
The Hailo PCIe driver (`hailo1x_pci`) is a **DKMS** module. When the Pi's kernel is upgraded
(e.g. `6.8.0-1057` → `6.8.0-1060`), DKMS may not auto-rebuild it, so on reboot there's **no
`/dev/hailo0`** and every Hailo call fails with *"Failed to create VDevice"*.

**Symptoms:** `ls /dev/hailo*` → not found; `lsmod | grep hailo` → empty; offline LLM 500s.

**Fix (needs sudo — run on the Pi):**
```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)   # headers for the NEW kernel
sudo dkms autoinstall                               # rebuild hailo1x_pci for it
sudo modprobe hailo1x_pci                            # load it now
ls -la /dev/hailo*                                   # expect /dev/hailo0
hailortcli fw-control identify                       # expect: Device Architecture: HAILO10H
```
Then restart Stella: `sudo -n systemctl restart airobot` (the NPU warmup will preload qwen).

**To prevent silent breakage** on future kernel updates, consider holding the kernel:
`sudo apt-mark hold linux-image-raspi linux-headers-raspi` (optional).

## Quick health checks
```bash
ls /dev/hailo*                                        # device present?
lsmod | grep hailo                                    # driver loaded?
curl -s localhost:8000/api/tags                       # models available?
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-instruct:1.5b","messages":[{"role":"user","content":"hi"}],"stream":false}'
```
