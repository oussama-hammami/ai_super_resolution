# Real-time AI Super Resolution (FSRCNN)

Real-time ×4 video upscaling with [FSRCNN](FSRCNN_Pytorch/README.md) on an NVIDIA GPU.
A 540×480 video is upscaled to 2160×1920 and displayed live, directly from GPU memory.

**Result: 12.4 FPS → 49.5 FPS** on an RTX 2060 (≈4× faster than the original pipeline).

---

## Setup

Tested on Ubuntu (GNOME, Wayland), NVIDIA GeForce RTX 2060 (6 GB), driver 595.91,
Python 3.14, PyTorch 2.11 + CUDA 12.8.

### 1. System packages

`torch.compile` (Triton) builds C helpers on first run and needs a C compiler and the
Python headers:

```bash
sudo apt install build-essential python3.14-dev   # match your Python version
```

Without them, the run fails with `Failed to find C compiler`. To run without compiling
instead: `TORCH_COMPILE_DISABLE=1 python test_pipeline_threading.py`.

### 2. Python environment

```bash
python3 -m venv .ai_super_res
source .ai_super_res/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch from the CUDA 12.8 wheel index (NVIDIA driver ≥ 570).

### 3. NVIDIA driver check

After a driver update, **reboot** before running. If the kernel module and the driver
libraries have different versions, OpenGL silently falls back to software rendering
(`llvmpipe`) and CUDA-OpenGL interop fails with `cudaErrorInvalidGraphicsContext`.
Symptom: `nvidia-smi` prints `Driver/library version mismatch`.

---

## Usage

```bash
source .ai_super_res/bin/activate
python test_pipeline_threading.py     # optimized pipeline, press q to quit
```

The average FPS (excluding the first 30 warm-up frames) is printed at the end. The first
run after a reboot takes longer because `torch.compile` recompiles the model.

Input video, device and checkpoint are set in [config.py](config.py).

| Script | Purpose |
|---|---|
| [test_pipeline_threading.py](test_pipeline_threading.py) | Optimized multithreaded pipeline with GPU display |
| [test_pipeline.py](test_pipeline.py) | Simple single-threaded loop (`cv2.imshow`) |
| [FSRCNN_Pytorch/demo.py](FSRCNN_Pytorch/demo.py) | Upscale a single image and display it |

### Running the original (pre-optimization) version

```bash
git worktree add ../sr_baseline 9cb2898
cd ../sr_baseline && python test_pipeline.py
git worktree remove ../sr_baseline    # when done
```

---

## Pipeline

```
 Thread 1        Thread 2              Thread 3                        Thread 4
 read frame  ->  preprocess (CPU)  ->  batched inference (GPU)     ->  display (GPU)
 cv2             blur, RGB->YCbCr      FSRCNN x4, fp16, compiled       CUDA -> OpenGL PBO
                                       + YCbCr->RGB, CHW->HWC (GPU)    -> texture -> window
```

Stages communicate through bounded queues (30 frames). Upscaled frames never leave the
GPU: the display thread copies each frame device-to-device into an OpenGL Pixel Buffer
Object (PBO) registered with CUDA, then draws it as a texture.

---

## Optimizations

| # | Optimization | Where |
|---|---|---|
| 1 | Multithreaded pipeline (read / preprocess / infer / display) with batching | `test_pipeline_threading.py` |
| 2 | fp16 inference (`torch.autocast`) + `no_grad` | `FSRCNN_Pytorch/model.py` |
| 3 | `torch.compile` of the model | `FSRCNN_Pytorch/model.py` |
| 4 | Post-processing (denorm, YCbCr→RGB, CHW→HWC) moved to the GPU, in the inference thread | `test_pipeline_threading.py` |
| 5 | Zero-copy display: persistent `GLDisplay` (CUDA-OpenGL interop) replaces `.cpu()` + numpy + `cv2.imshow` | `FSRCNN_Pytorch/utils/display.py` |

Also fixed: pressing `q` could deadlock the pipeline (the display thread stopped draining
its queue, so upstream threads blocked on a full queue).

---

## Benchmark results

540×480 input → 2160×1920 output, 1613 frames, RTX 2060. FPS is the steady-state average
over the whole video (warm-up excluded).

| Version | Avg FPS |
|---|---|
| Original, single thread (commit `9cb2898`) | 12.4 |
| + fp16, single thread | 15.5 |
| + multithreading, no `torch.compile` | 21.8 |
| + `torch.compile` | 19.6 |
| + GPU post-processing, pinned download, `cv2.imshow` | 36.3 |
| **+ zero-copy OpenGL display (current)** | **49.5** |

`torch.compile` alone did not raise FPS: the display thread was the bottleneck
(~42 ms/frame for colour conversion, download, numpy copy and `cv2.imshow`).
Moving that work to the GPU is what unlocked the gains.

### Per-stage timings (before the display fixes, with `torch.compile`)

| Stage | ms / frame |
|---|---|
| `cap.read` | 0.3 |
| Preprocess on CPU (blur + YCbCr) | 10.9 |
| Inference, batch 1 (fp16) | 12.5 |
| Inference, batch 4 (per frame) | 10.1 |
| Post-process on GPU | 2.8 |
| `.cpu()` download | 4.6 |
| numpy transpose + BGR copy | 20.7 |
| `cv2.imshow` | 14.1 |

### Higher input resolution

With the sample video resized to a 1920×1080 input, the ×4 output is 7680×4320 (8K),
8× the pixels of the default output. It ran at **5.4 FPS**, using 4.3 / 6 GB of GPU
memory. The default configuration uses the 540×480 video.

---

## Known limitations and next steps

- **No window title bar on GNOME Wayland**: GLFW's libdecor plugin fails to load, so the
  window has no decorations and the live FPS in the title is not visible. Press `q` to
  quit. Running through XWayland (`PYGLFW_LIBRARY_VARIANT=x11 PYOPENGL_PLATFORM=glx`)
  restores the title bar but drops to ~32 FPS.
- **Display is capped by vsync** at the monitor refresh rate (60 Hz here).
- **Preprocessing still runs on the CPU** (~11 ms/frame). Moving blur and RGB→YCbCr to
  the GPU is the next optimization.
- Further options: CUDA graphs (`torch.compile(mode="reduce-overhead")`), TensorRT fp16 export.

---

## Credits

Model and training code: [FSRCNN-Pytorch](FSRCNN_Pytorch/README.md) by VuNguyenNhatThanh
(MIT License, see [FSRCNN_Pytorch/LICENSE](FSRCNN_Pytorch/LICENSE)).
