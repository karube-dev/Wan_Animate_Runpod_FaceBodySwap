# Wan Animate RunPod Face/Body Swap

**ComfyUI + WanAnimate + ReActor face swap — deployed as a RunPod Serverless worker.**

This project builds on the upstream [wlsdml1114/Wan_Animate_Runpod_hub](https://github.com/wlsdml1114/Wan_Animate_Runpod_hub) baseline and adds **face swap** capabilities using [ReActor](https://github.com/Gourieff/comfyui-reactor-node) + [InsightFace](https://github.com/deepinsight/insightface).

---

## What this does

1. Takes a **subject image** (person A) and a **reference motion video** (person B's movement).
2. Generates an animated video of person A performing person B's movements using **WanAnimate (Wan2.2-Animate-14B)**.
3. Optionally swaps the face in the result to a **desired identity** using ReActor:
   - **`pre` mode** — swap the face on the input image *before* WanAnimate generation. CLIP Vision sees the swapped face, improving identity consistency throughout.
   - **`post` mode** — swap the face on every output frame *after* WanAnimate generation. More reliable enforcement of the target identity.
   - **`both` mode** (default) — applies both pre- and post-swap for best results.

> **Note:** This implementation focuses on **face swap**. Full body replacement (e.g., replacing the entire body silhouette) is not directly supported — the pipeline transfers body motion from the reference video and swaps the facial identity. This is explicitly a face-swap oriented pipeline, not a full-body clone.

---

## Architecture

```
swap_image ──────────────────────────────────────────┐
                                                      ↓
input_image ──→ [ReActor Pre-Swap] ──→ swapped_image  │
                                           ↓           │
reference_video ──→ WanAnimate (Wan2.2-14B) ──→ frames │
                                           ↓           │
                    [ReActor Post-Swap (all frames)] ←─┘
                                           ↓
                               Final video (MP4)
```

---

## Deploying to RunPod Serverless

### Prerequisites

- A [RunPod](https://runpod.io) account
- This repository on GitHub (public or private)
- A GPU with **≥ 48 GB VRAM** recommended (A6000/L40S/A100 80GB). A 24 GB GPU may work without face swap but is tight with all models loaded.

### Step 1 — Connect GitHub to RunPod

1. Open [RunPod Console](https://www.runpod.io/console) → **Settings** → **Connections** → **GitHub** → **Connect**
2. Authorize RunPod to access your GitHub account.
3. If using a private repository, select it explicitly in the repository access list.

### Step 2 — Create a Serverless Endpoint

1. Go to **Serverless** → **New Endpoint**
2. Select **Import Git Repository**
3. Choose this repository and the `main` branch
4. Set **Dockerfile path** to `Dockerfile` (default)
5. Click **Next**

> **RunPod builds the Docker image automatically** when you create or update the endpoint via a GitHub release. You do **not** need to build or push a Docker image yourself.

### Step 3 — Configure the Endpoint

| Setting | Recommended value |
|---|---|
| **GPU** | A6000 (48 GB) or A100 80 GB |
| **Active workers** | 0 (scale to zero when idle) |
| **Max workers** | 1–2 |
| **Execution timeout** | 900–1800 seconds |
| **Container disk** | 80 GB |

### Step 4 — Trigger a Build

RunPod builds the image when you **create a GitHub Release** on the target branch:

1. In GitHub: **Releases** → **Draft a new release** → set a tag (e.g., `v1.0.0`) → **Publish release**
2. RunPod detects the release and starts a build. Monitor it in **Builds** tab of your endpoint.

### Step 5 — Test the endpoint

Use the **Requests** tab in RunPod Console or send a POST request:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d @request.json
```

---

## API Reference

### Required inputs

| Field | Type | Description |
|---|---|---|
| `prompt` | string | Text describing the desired output motion/scene |
| `image_url` / `image_path` / `image_base64` | string | Subject image (person whose body will animate) |
| `video_url` / `video_path` / `video_base64` | string | Reference motion video |
| `seed` | int | Random seed |
| `width` | int | Output width (e.g., 832) |
| `height` | int | Output height (e.g., 480) |
| `fps` | int | Frames per second (e.g., 16) |
| `cfg` | float | CFG scale (1.0 recommended) |

### Optional inputs

| Field | Type | Default | Description |
|---|---|---|---|
| `steps` | int | 4 | Diffusion steps |
| `negative_prompt` | string | — | Negative text prompt |
| `face_swap_enabled` | bool | `false` | Enable ReActor face swap |
| `swap_image_url` / `swap_image_path` / `swap_image_base64` | string | — | **Required when `face_swap_enabled=true`.** The face identity to swap in. |
| `swap_mode` | string | `"both"` | `"pre"`, `"post"`, or `"both"` |
| `face_restore_model` | string | `"GFPGANv1.4.pth"` | Face restoration model. Options: `"GFPGANv1.4.pth"`, `"codeformer.pth"` |
| `face_restore_visibility` | float | `1.0` | Face restoration strength (0.0–1.0) |
| `codeformer_weight` | float | `0.5` | CodeFormer fidelity weight when using CodeFormer (0.0–1.0) |
| `mode` | string | `"replace"` | Workflow mode: `"replace"` or `"animate"` |

---

## Example Request Payloads

### Basic animation (no face swap)

```json
{
  "input": {
    "prompt": "A person walking naturally in a park",
    "image_url": "https://example.com/subject.jpg",
    "video_url": "https://example.com/motion_reference.mp4",
    "seed": 12345,
    "width": 832,
    "height": 480,
    "fps": 16,
    "cfg": 1.0,
    "steps": 4
  }
}
```

### Face swap — post mode (replace faces in generated video only)

```json
{
  "input": {
    "prompt": "A person walking naturally",
    "image_url": "https://example.com/subject.jpg",
    "video_url": "https://example.com/motion_reference.mp4",
    "seed": 42,
    "width": 832,
    "height": 480,
    "fps": 16,
    "cfg": 1.0,
    "steps": 4,
    "face_swap_enabled": true,
    "swap_image_url": "https://example.com/desired_face.jpg",
    "swap_mode": "post",
    "face_restore_model": "GFPGANv1.4.pth",
    "face_restore_visibility": 1.0
  }
}
```

### Face swap — both modes (highest identity consistency)

```json
{
  "input": {
    "prompt": "A person dancing energetically",
    "image_url": "https://example.com/subject.jpg",
    "video_url": "https://example.com/dance_reference.mp4",
    "seed": 99,
    "width": 832,
    "height": 480,
    "fps": 16,
    "cfg": 1.0,
    "steps": 6,
    "face_swap_enabled": true,
    "swap_image_url": "https://example.com/target_face.jpg",
    "swap_mode": "both",
    "face_restore_model": "GFPGANv1.4.pth",
    "face_restore_visibility": 1.0,
    "codeformer_weight": 0.5
  }
}
```

### Face swap with base64 image

```json
{
  "input": {
    "prompt": "A person running",
    "image_base64": "<base64-encoded-subject-image>",
    "video_url": "https://example.com/motion.mp4",
    "seed": 7,
    "width": 832,
    "height": 480,
    "fps": 16,
    "cfg": 1.0,
    "face_swap_enabled": true,
    "swap_image_base64": "<base64-encoded-face-image>",
    "swap_mode": "both"
  }
}
```

### Response

A successful response returns the generated video as a base64-encoded MP4:

```json
{
  "id": "...",
  "status": "COMPLETED",
  "output": {
    "video": "<base64-encoded-mp4>"
  }
}
```

---

## Caveats and Limitations

### GPU size requirements

| Configuration | Min VRAM |
|---|---|
| WanAnimate only (no face swap) | 24 GB (with `blocks_to_swap=25`) |
| WanAnimate + face swap | 48 GB recommended |
| WanAnimate + face swap (full quality) | 80 GB ideal |

The Wan2.2-Animate-14B model is large. The current configuration uses `blocks_to_swap=25` to offload model blocks to CPU when VRAM is insufficient. On GPUs with more VRAM (48–80 GB), you can reduce `blocks_to_swap` in the workflow JSON to improve inference speed.

### Model download time

The Dockerfile downloads approximately **30–40 GB** of models during the build:
- Wan2.2-Animate-14B FP8 (~14 GB)
- LoRA models (~4 GB total)
- InsightFace inswapper_128 (~500 MB)
- GFPGAN / CodeFormer face restoration (~300 MB)
- Detection models (YOLOv10m, ViTPose)

The first RunPod build will take **20–40 minutes** depending on Hugging Face download speeds.

### Video quality and latency

- Generation at 832×480, 16fps, 4 steps typically takes **5–15 minutes** on an A6000.
- Increasing `steps` to 6–8 improves quality but adds proportional time.
- Face swap post-processing adds ~1–2 minutes for a typical 16-frame video.

### Face swap vs. body swap

- This pipeline does **face swap** using ReActor + InsightFace inswapper_128.
- **Body shape** is determined by the subject image and the reference motion — it is NOT replaced by the swap source's body.
- For best results, use a swap source image where the face is clearly visible, frontal, and well-lit.
- The `pre` mode influences what CLIP Vision encodes, which can subtly affect body appearance, but this is not equivalent to full body swap.

### Multiple faces

- By default, only the first detected face (index 0) is swapped.
- If the video contains multiple people, all detected faces may be affected. Use `detect_gender_input` and `detect_gender_source` to filter if needed (passed via custom workflow modifications).

---

## Custom Nodes Used

| Node | Purpose |
|---|---|
| [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) | WanAnimate model loading, sampling, decoding |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Utility nodes (resize, constants, etc.) |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | Video loading and combining |
| [ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess) | Pose/face detection preprocessing |
| [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) | SAM2 segmentation for masks |
| [IntelligentVRAMNode](https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode) | Smart VRAM management |
| [ComfyUI-AdaptiveWindowSize](https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize) | Adaptive frame window processing |
| [comfyui-reactor-node](https://github.com/Gourieff/comfyui-reactor-node) | **Face swap using InsightFace** |

---

## License

Based on [wlsdml1114/Wan_Animate_Runpod_hub](https://github.com/wlsdml1114/Wan_Animate_Runpod_hub). Please review upstream licenses for all custom nodes and models used.
