import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    Check if input data is a Base64 string; if so, save as a file and return the path.
    If it is a plain path string, return it as-is.
    """
    if not isinstance(data_input, str):
        return data_input

    try:
        decoded_data = base64.b64decode(data_input)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        print(f"✅ Saved Base64 input to '{file_path}'.")
        return file_path

    except (binascii.Error, ValueError):
        print(f"➡️ '{data_input}' treated as a file path.")
        return data_input

def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                with open(video['fullpath'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append(video_data)
        output_videos[node_id] = videos_output

    return output_videos

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def process_input(input_data, temp_dir, output_filename, input_type):
    """Process input data and return a local file path."""
    if input_type == "path":
        logger.info(f"📁 Path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info(f"🔢 Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def download_file_from_url(url, output_path):
    """Download a file from a URL using wget."""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✅ Downloaded from URL: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Download timed out")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decode Base64 data and save to file."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Saved Base64 input to '{file_path}'.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")


def apply_face_swap_nodes(prompt, swap_image_path, swap_mode, face_restore_model, face_restore_visibility, codeformer_weight):
    """
    Inject ReActor face swap nodes into the workflow prompt.

    swap_mode:
      "pre"  - swap the identity on the input image BEFORE WanAnimate generation.
               The CLIP Vision encoder will see the swapped face, improving
               consistency throughout the generated video.
      "post" - apply face swap on every output frame AFTER WanAnimate generation.
               More reliable face identity enforcement on the final video.
      "both" - apply both pre and post swap for highest identity consistency.

    Node layout:
      300 = LoadImage (swap source, pre-swap path)
      301 = ReActorFaceSwap (pre-swap: input node 57 → output fed into node 64)
      310 = LoadImage (swap source, post-swap path — same file, separate node)
      311 = ReActorFaceSwap (post-swap: input node 194 → output fed into node 30)
    """
    reactor_defaults = {
        "swap_model": "inswapper_128.onnx",
        "facedetection": "retinaface_resnet50",
        "face_restore_model": face_restore_model,
        "face_restore_visibility": face_restore_visibility,
        "codeformer_weight": codeformer_weight,
        "detect_gender_input": "no",
        "detect_gender_source": "no",
        "input_faces_index": "0",
        "source_faces_index": "0",
        "console_log_level": 1,
    }

    if swap_mode in ("pre", "both"):
        # Node 300: load the swap-source image
        prompt["300"] = {
            "inputs": {"image": swap_image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Face Swap Source Image (Pre)"},
        }
        # Node 301: swap the face on the input image before WanAnimate
        prompt["301"] = {
            "inputs": {
                **reactor_defaults,
                "enabled": True,
                "input_image": ["57", 0],   # original input image
                "source_image": ["300", 0], # swap-source face
            },
            "class_type": "ReActorFaceSwap",
            "_meta": {"title": "Pre-Process Face Swap (ReActor)"},
        }
        # Redirect ImageResizeKJv2 (node 64) to use the swapped image
        prompt["64"]["inputs"]["image"] = ["301", 0]
        logger.info("✅ Pre-swap ReActor node injected (node 301).")

    if swap_mode in ("post", "both"):
        # Node 310: load the swap-source image (separate LoadImage node)
        prompt["310"] = {
            "inputs": {"image": swap_image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Face Swap Source Image (Post)"},
        }
        # Node 311: apply face swap on all output frames from WanAnimate
        prompt["311"] = {
            "inputs": {
                **reactor_defaults,
                "enabled": True,
                "input_image": ["194", 0],  # WanAnimate generated frames
                "source_image": ["310", 0], # swap-source face
            },
            "class_type": "ReActorFaceSwap",
            "_meta": {"title": "Post-Process Face Swap (ReActor)"},
        }
        # Redirect VHS_VideoCombine (node 30) to use the swapped frames
        prompt["30"]["inputs"]["images"] = ["311", 0]
        logger.info("✅ Post-swap ReActor node injected (node 311).")

    return prompt


def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    # ── Input image ──────────────────────────────────────────────────────────
    image_path = None
    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")

    # ── Reference motion video ────────────────────────────────────────────────
    video_path = None
    if "video_path" in job_input:
        video_path = process_input(job_input["video_path"], task_id, "input_video.mp4", "path")
    elif "video_url" in job_input:
        video_path = process_input(job_input["video_url"], task_id, "input_video.mp4", "url")
    elif "video_base64" in job_input:
        video_path = process_input(job_input["video_base64"], task_id, "input_video.mp4", "base64")

    # ── Face/body swap inputs ─────────────────────────────────────────────────
    face_swap_enabled = job_input.get("face_swap_enabled", False)
    swap_image_path = None

    if face_swap_enabled:
        if "swap_image_path" in job_input:
            swap_image_path = process_input(
                job_input["swap_image_path"], task_id, "swap_source.jpg", "path")
        elif "swap_image_url" in job_input:
            swap_image_path = process_input(
                job_input["swap_image_url"], task_id, "swap_source.jpg", "url")
        elif "swap_image_base64" in job_input:
            swap_image_path = process_input(
                job_input["swap_image_base64"], task_id, "swap_source.jpg", "base64")

        if swap_image_path is None:
            raise Exception(
                "face_swap_enabled=true but no swap image provided. "
                "Supply swap_image_path, swap_image_url, or swap_image_base64."
            )

    swap_mode = job_input.get("swap_mode", "both")
    if swap_mode not in ("pre", "post", "both"):
        raise Exception(f"Invalid swap_mode '{swap_mode}'. Must be 'pre', 'post', or 'both'.")

    face_restore_model = job_input.get("face_restore_model", "GFPGANv1.4.pth")
    face_restore_visibility = float(job_input.get("face_restore_visibility", 1.0))
    codeformer_weight = float(job_input.get("codeformer_weight", 0.5))

    # ── Validate required inputs ──────────────────────────────────────────────
    if image_path is None:
        raise Exception("Image input is required. Provide image_path, image_url, or image_base64.")
    if video_path is None:
        raise Exception("Video input is required. Provide video_path, video_url, or video_base64.")

    # ── Load workflow ─────────────────────────────────────────────────────────
    prompt = None
    check_coord = job_input.get("points_store", None)

    if face_swap_enabled:
        # Use the dedicated face-swap workflow (has both SAM2 + ReActor placeholders)
        prompt = load_workflow('/wananimate_faceswap_api.json')
    elif check_coord is None:
        if job_input.get("mode", "replace") == "animate":
            prompt = load_workflow('/newWanAnimate_noSAM_animate_api.json')
        else:
            prompt = load_workflow('/newWanAnimate_noSAM_api.json')
    else:
        if job_input.get("mode", "replace") == "animate":
            prompt = load_workflow('/newWanAnimate_noSAM_animate_api.json')
        else:
            prompt = load_workflow('/newWanAnimate_noSAM_api.json')

    if prompt is None:
        raise RuntimeError("Failed to load workflow prompt.")

    # ── Populate common workflow parameters ───────────────────────────────────
    prompt["57"]["inputs"]["image"] = image_path
    prompt["63"]["inputs"]["video"] = video_path
    prompt["63"]["inputs"]["force_rate"] = job_input["fps"]
    prompt["30"]["inputs"]["frame_rate"] = job_input["fps"]
    prompt["65"]["inputs"]["positive_prompt"] = job_input["prompt"]
    if "negative_prompt" in job_input:
        prompt["65"]["inputs"]["negative_prompt"] = job_input["negative_prompt"]
    prompt["27"]["inputs"]["seed"] = job_input["seed"]
    prompt["27"]["inputs"]["cfg"] = job_input["cfg"]
    prompt["27"]["inputs"]["steps"] = job_input.get("steps", 4)
    prompt["150"]["inputs"]["value"] = job_input["width"]
    prompt["151"]["inputs"]["value"] = job_input["height"]

    # ── Point-based SAM coordinates (optional) ────────────────────────────────
    if check_coord is not None and "107" in prompt:
        prompt["107"]["inputs"]["points_store"] = job_input["points_store"]
        prompt["107"]["inputs"]["coordinates"] = job_input["coordinates"]
        prompt["107"]["inputs"]["neg_coordinates"] = job_input["neg_coordinates"]

    # ── Inject face swap nodes if enabled ─────────────────────────────────────
    if face_swap_enabled and swap_image_path:
        prompt = apply_face_swap_nodes(
            prompt,
            swap_image_path,
            swap_mode,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
        )
        logger.info(f"🎭 Face swap enabled: mode={swap_mode}, restore={face_restore_model}")

    # ── Connect to ComfyUI ────────────────────────────────────────────────────
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Connecting to ComfyUI at {http_url}")

    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection OK (attempt {http_attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connection failed (attempt {http_attempt + 1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to ComfyUI. Is it running?")
            time.sleep(1)

    ws = websocket.WebSocket()
    max_ws_attempts = int(180 / 5)
    for attempt in range(max_ws_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connection failed (attempt {attempt + 1}/{max_ws_attempts}): {e}")
            if attempt == max_ws_attempts - 1:
                raise Exception("WebSocket connection timed out (3 minutes).")
            time.sleep(5)

    videos = get_videos(ws, prompt)
    ws.close()

    for node_id in videos:
        if videos[node_id]:
            return {"video": videos[node_id][0]}

    return {"error": "No video found in outputs."}


runpod.serverless.start({"handler": handler})
