# Hailo‑10H vision offload — setup guide (2026‑10‑31)

Goal: move Stella's heavy vision (object detection, and ideally face + depth) OFF the Pi CPU and ONTO the
Hailo‑10H, while keeping the offline LLM. Written for you to read **before** committing.

## What we have today (verified on the Pi)
- **Hailo‑10H**, firmware **5.1.1**, `/dev/hailo0` live. It is **already running the offline LLM** via
  `hailo-ollama` (apt: `h10-hailort`, `h10-hailort-pcie-driver`, `hailo-gen-ai-model-zoo`, all 5.1.1). Not idle.
- The venv's Python runtime is `hailort 4.23.0` (PyPI) — **mismatched**, can't drive the NPU.
- **No vision `.hef`** installed (the gen‑ai model zoo is LLMs only: qwen2.5/llama3.2/deepseek).

## What a free Hailo Developer Zone account unlocks
Register at **hailo.ai → Developer Zone** (free, email only). It gives:
1. **HailoRT** runtime + the matched **`pyhailort` wheel** (Python API to run `.hef` on the chip).
2. **Vision Model Zoo `.hef`** compiled for **Hailo‑10H**: YOLOv8/YOLO11 (detection), SCRFD/RetinaFace + ArcFace
   (faces), monocular **depth** (for the eyes/arm "how far" goal). Hailo‑8 `.hef` do NOT run on the 10H.
3. The **Dataflow Compiler** (x86 Linux) — only needed to compile a custom ONNX (e.g. an NVIDIA model) to `.hef`.

## The catch — a stack upgrade is required (the main risk)
Prebuilt Hailo‑10H vision `.hef` require **HailoRT 5.2.0+**; on 5.1.1 they fail with `HAILO_NOT_IMPLEMENTED`
(see sources). So we must upgrade the Hailo stack from **5.1.1 → 5.3.0** (firmware + PCIe driver + runtime +
pyhailort). **Risk:** `hailo-ollama` (your working LLM) runs on the 5.1.1 stack; after the upgrade it must be
re‑matched/updated, and the offline brain could be down until it is. We mitigate by backing up and testing the
LLM right after the upgrade before touching vision.

## Plan (what I do once you have the account)
1. **You:** create the free account; either download the files I list, or paste me the download links.
2. **Stack upgrade (Pi, your sudo):** back up the current Hailo packages, then upgrade to 5.3.0 — easiest via
   Hailo's RPi app layer **`hailo-apps`** (`hailo_installer_python.sh hailo10h --hailort-version 5.3.0`) plus the
   5.3.0 PCIe driver `.deb` from Hailo's dev‑public server; install `pyhailort` 5.3.0 into the venv.
3. **Re‑verify the LLM:** confirm `hailo-ollama` still answers (upgrade/repoint it if needed). The brain comes first.
4. **Models:** download Hailo‑10H `.hef` (hailo‑apps' resource downloader fetches most): YOLOv8s/YOLO11 detection,
   SCRFD face, a depth model (scdepth/fast‑depth).
5. **Integrate:** rewrite `parts_used/hailo_10h.py` to run detection (+ optionally face + depth) on `/dev/hailo0`
   via pyhailort, replacing the OpenCV‑DNN CPU path. One NPU serves BOTH the LLM and vision, so vision runs
   continuously and LLM queries **time‑share** (or fall back to cloud while vision is hot).
6. **Verify:** Pi CPU drops sharply; detection + depth work; the LLM still answers. Update the architecture doc.

## What I need from you
- The **free Hailo Developer Zone account** (or the downloaded files / links).
- A **go‑ahead to upgrade the Hailo stack to 5.3.0**, accepting the LLM may need re‑matching (I back up first).

## Lower‑risk alternative (switch anytime)
Keep 5.1.1 + the LLM exactly as‑is, and I reduce the Pi CPU vision load now (face‑recog/liveness/detection
cadence) so she's responsive without touching the NPU. Less dramatic, zero risk to the brain.

## NVIDIA note
NVIDIA's free models don't run on the Hailo directly (no NVIDIA HW here). You already use NVIDIA's **cloud** (NIM)
as the scene VLM. On‑device, an NVIDIA ONNX would be Hailo‑compiled like any other model (DFC, needs the account).

## Sources
- Hailo Model Zoo (Hailo‑10H, DFC v5.x): https://github.com/hailo-ai/hailo_model_zoo
- Hailo‑10H on RPi5 (Ubuntu) install + verify: https://github.com/martinlars/hailo-10h-ubuntu-rpi5
- Hailo‑10H RPi5 builder (latest drivers/firmware/runtime): https://github.com/abagonhishead/hailo-10h-rpi5-builder
- Upgrading to HailoRT 5.2.0 step‑by‑step (RPi + hailo‑apps): https://community.hailo.ai/t/upgrading-to-hailort-5-2-0-step-by-step-raspberry-pi-hailo-apps/19006
- YOLO on RPi5 Hailo‑10H (HEFs need 5.2.0; 5.1.1 = HAILO_NOT_IMPLEMENTED): https://github.com/MichalAnatolSkora/yolo-on-rpi5-hailo10h
- Ultralytics Hailo export (DFC v5.x for 10H): https://docs.ultralytics.com/integrations/hailo
- Raspberry Pi AI software docs: https://www.raspberrypi.com/documentation/computers/ai.html
