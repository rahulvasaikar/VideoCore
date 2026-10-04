"""FastAPI Backend for VideoCore Web Application."""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from videocore.core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from videocore.core.ffmpeg import detect_available_gpu_encoders, get_ffmpeg_path, get_ffprobe_path
from videocore.core.probe import are_streams_compatible, probe_video
from videocore.tools.animator.engine import AnimationConfig, AnimatorEngine
from videocore.tools.animator.models import (
    ensure_all_anime_models,
    get_animegan_model_path,
)
from videocore.tools.enhancer.engine import EnhanceConfig, EnhancerEngine
from videocore.tools.enhancer.models import (
    ensure_all_models,
    get_gfpgan_model,
    get_realesrgan_bin,
    get_rife_bin,
    get_yunet_model,
)
from videocore.tools.face_studio.analyzer import DetectedFace
from videocore.tools.face_studio.engine import FaceStudioConfig, FaceStudioEngine
from videocore.tools.face_studio.models import ensure_face_models, get_sface_model
from videocore.tools.joiner.cli import RESOLUTION_PRESETS, discover_folder_videos, parse_resolution
from videocore.tools.joiner.engine import JoinConfig, JoinMode, VideoJoiner
from videocore.ui.dialogs import (
    open_in_file_explorer,
    play_video,
    select_folder,
    select_media_file,
    select_save_file,
    select_video_files,
)

app = FastAPI(title="VideoCore API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global Job & Cache Store ---
class JobState:
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.status = "running"  # running, completed, error
        self.progress = 0.0
        self.stage = "Initializing..."
        self.detail = ""
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "detail": self.detail,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
        }


JOBS: Dict[str, JobState] = {}
FACE_CLUSTERS_CACHE: Dict[str, List[DetectedFace]] = {}


def bgr_to_base64_jpeg(img_bgr: np.ndarray, quality: int = 85) -> str:
    """Converts a BGR image array to a data URL string."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def validate_path(p: str) -> Path:
    path = Path(p).resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {p}")
    return path


# --- System Info & Hardware Detection ---
@app.get("/api/system")
def api_system_info():
    try:
        gpu_map = detect_available_gpu_encoders()
        return {
            "success": True,
            "has_nvenc": gpu_map.get("nvenc", False),
            "has_qsv": gpu_map.get("qsv", False),
            "has_amf": gpu_map.get("amf", False),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/models/status")
def api_models_status():
    try:
        esrgan_ok = get_realesrgan_bin() is not None
        rife_ok = get_rife_bin() is not None
        yunet_ok = get_yunet_model() is not None
        gfpgan_ok = get_gfpgan_model() is not None
        sface_ok = get_sface_model() is not None
        hayao_ok = get_animegan_model_path("hayao") is not None
        shinkai_ok = get_animegan_model_path("shinkai") is not None
        paprika_ok = get_animegan_model_path("paprika") is not None
        all_present = (
            esrgan_ok and rife_ok and yunet_ok and gfpgan_ok and sface_ok and hayao_ok and shinkai_ok and paprika_ok
        )
        return {
            "success": True,
            "esrgan": esrgan_ok,
            "rife": rife_ok,
            "yunet": yunet_ok,
            "gfpgan": gfpgan_ok,
            "sface": sface_ok,
            "hayao": hayao_ok,
            "shinkai": shinkai_ok,
            "paprika": paprika_ok,
            "all_present": all_present,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/models/download")
def api_models_download():
    job_id = str(uuid.uuid4())
    job = JobState(job_id, "models_download")
    JOBS[job_id] = job

    def worker():
        try:
            job.stage = "Downloading AI Models..."
            job.progress = 20.0
            ensure_all_models()
            job.progress = 60.0
            ensure_all_anime_models()
            job.progress = 90.0
            ensure_face_models()
            job.progress = 100.0
            job.stage = "All models downloaded successfully!"
            job.status = "completed"
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


# --- Native File Dialogs ---
@app.get("/api/dialog/file")
def api_select_file(title: str = "Select File"):
    path = select_media_file(title=title)
    return {"path": path}


@app.get("/api/dialog/folder")
def api_select_folder(title: str = "Select Folder"):
    path = select_folder(title=title)
    return {"path": path}


@app.get("/api/dialog/files")
def api_select_files(title: str = "Select Files"):
    paths = select_video_files(title=title)
    return {"paths": paths}


@app.get("/api/dialog/save")
def api_select_save(title: str = "Save As", default_name: str = "output.mp4"):
    path = select_save_file(title=title, default_name=default_name)
    return {"path": path}


# --- Actions ---
class PathRequest(BaseModel):
    path: str


@app.post("/api/action/play")
def api_play_video(req: PathRequest):
    try:
        path = validate_path(req.path)
        play_video(path)
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/action/explore")
def api_open_explorer(req: PathRequest):
    try:
        path = validate_path(req.path)
        open_in_file_explorer(path)
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


# --- Probing & Media Discovery ---
@app.post("/api/probe")
def api_probe(req: PathRequest):
    try:
        path = validate_path(req.path)
        meta = probe_video(path)
        ext = path.suffix.lower()
        is_image = ext in IMAGE_EXTENSIONS
        is_video = ext in VIDEO_EXTENSIONS

        pv = meta.primary_video
        pa = meta.primary_audio

        return {
            "success": True,
            "path": str(path),
            "filename": path.name,
            "is_image": is_image,
            "is_video": is_video,
            "duration": meta.duration or 0.0,
            "format_name": meta.format_name,
            "primary_video": {
                "width": pv.width,
                "height": pv.height,
                "resolution": f"{pv.width}x{pv.height}",
                "fps": round(pv.fps, 2),
                "codec": pv.codec_name,
                "bitrate": pv.bitrate,
                "pix_fmt": pv.pix_fmt,
            } if pv else None,
            "primary_audio": {
                "codec": pa.codec_name,
                "sample_rate": pa.sample_rate,
                "channels": pa.channels,
            } if pa else None,
            "video_streams": [
                {
                    "codec": vs.codec_name,
                    "resolution": f"{vs.width}x{vs.height}",
                    "fps": round(vs.fps, 2),
                    "pix_fmt": vs.pix_fmt,
                    "bitrate": vs.bitrate,
                }
                for vs in meta.video_streams
            ],
            "audio_streams": [
                {
                    "codec": as_.codec_name,
                    "sample_rate": as_.sample_rate,
                    "channels": as_.channels,
                }
                for as_ in meta.audio_streams
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/discover")
def api_discover(req: PathRequest):
    try:
        path = validate_path(req.path)
        if not path.is_dir():
            raise ValueError("Path is not a directory")
        paths = discover_folder_videos(path)
        return {"success": True, "paths": [str(p.resolve()) for p in paths]}
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


# --- Media Streaming & Thumbnails ---
@app.get("/api/media/thumbnail")
def api_media_thumbnail(path: str = Query(...), time: float = Query(0.0)):
    try:
        p = validate_path(path)
        ext = p.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            img = cv2.imread(str(p))
        elif ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(time * fps))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise ValueError("Could not read frame")
            img = frame
        else:
            raise ValueError("Unsupported media type")

        # Resize if very large for fast web delivery
        h, w = img.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            raise RuntimeError("Encoding thumbnail failed")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/media/file")
def api_media_file(path: str = Query(...)):
    try:
        p = validate_path(path)
        return FileResponse(str(p))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# --- Jobs Status Endpoint ---
@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# --- TAB 1: JOINER EXECUTE ---
class JoinExecuteRequest(BaseModel):
    inputs: List[str]
    output: str
    mode: str = "auto"
    preset: str = "Match First Clip"
    target_fps: Optional[float] = None


@app.post("/api/join/execute")
def api_join_execute(req: JoinExecuteRequest):
    if not req.inputs:
        return JSONResponse(status_code=400, content={"error": "No input files provided"})
    if not req.output:
        return JSONResponse(status_code=400, content={"error": "Output destination is required"})

    job_id = str(uuid.uuid4())
    job = JobState(job_id, "join")
    JOBS[job_id] = job

    def worker():
        try:
            input_paths = [Path(p).resolve() for p in req.inputs]
            output_path = Path(req.output).resolve()

            # Parse custom resolution preset
            custom_res = None
            if req.preset != "Match First Clip":
                preset_lower = req.preset.lower()
                if "vertical" in preset_lower and "1080p" in preset_lower:
                    custom_res = RESOLUTION_PRESETS.get("vertical_1080p")
                elif "vertical" in preset_lower and "720p" in preset_lower:
                    custom_res = RESOLUTION_PRESETS.get("vertical_720p")
                elif "1080p" in preset_lower:
                    custom_res = RESOLUTION_PRESETS.get("1080p")
                elif "720p" in preset_lower:
                    custom_res = RESOLUTION_PRESETS.get("720p")
                elif "4k" in preset_lower:
                    custom_res = RESOLUTION_PRESETS.get("4k")

            join_mode = JoinMode(req.mode.lower())
            config = JoinConfig(
                inputs=input_paths,
                output=output_path,
                mode=join_mode,
                target_resolution=custom_res,
                target_fps=req.target_fps,
                gpu="auto",
            )
            joiner = VideoJoiner(config)

            def on_progress(fraction: float, speed: str, fps: str, time_str: str):
                pct = round(fraction * 100.0, 1)
                job.progress = pct
                job.stage = f"Joining: {int(pct)}% complete"
                job.detail = f"⚡ Time: {time_str} | Speed: {speed} | FPS: {fps}"

            def on_status(msg: str):
                job.stage = msg

            start_t = time.time()
            res = joiner.join(on_progress=on_progress, on_status=on_status)
            elapsed = time.time() - start_t

            job.progress = 100.0
            job.stage = "Join operation complete!"
            job.status = "completed"
            job.result = {
                "output_path": str(res.output_path),
                "duration": res.total_duration,
                "size_mb": round(res.output_size_bytes / (1024 * 1024), 2),
                "mode": res.mode_used.value.upper(),
                "encoder": res.encoder_used,
                "elapsed": round(elapsed, 1),
            }
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


# --- TAB 2: AI MEDIA ENHANCER ---
class EnhanceExecuteRequest(BaseModel):
    input: str
    output: str
    scale: int = 2
    model_name: str = "realesrgan-x4plus"
    face_restore: bool = True
    face_fidelity: float = 0.70
    interpolate: bool = False
    target_fps: Optional[float] = None
    video_codec: str = "h264_nvenc"


@app.post("/api/enhance/execute")
def api_enhance_execute(req: EnhanceExecuteRequest):
    try:
        inp = validate_path(req.input)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not req.output:
        return JSONResponse(status_code=400, content={"error": "Output destination is required"})

    job_id = str(uuid.uuid4())
    job = JobState(job_id, "enhance")
    JOBS[job_id] = job

    def worker():
        try:
            out_p = Path(req.output).resolve()
            config = EnhanceConfig(
                input_path=inp,
                output_path=out_p,
                scale=req.scale,
                model_name=req.model_name,
                face_restore=req.face_restore,
                face_fidelity=req.face_fidelity,
                interpolate=req.interpolate,
                target_fps=req.target_fps,
                video_codec=req.video_codec,
            )
            engine = EnhancerEngine(config)

            def stage_callback(stage_name: str, pct: float):
                job.progress = round(pct, 1)
                job.stage = f"{stage_name} ({pct:.0f}%)"

            start_t = time.time()
            res = engine.execute(stage_callback=stage_callback)
            elapsed = time.time() - start_t

            job.progress = 100.0
            job.stage = "AI Enhancement complete!"
            job.status = "completed"
            job.result = {
                "input": str(res.input_path),
                "output": str(res.output_path),
                "is_video": res.is_video,
                "orig_res": f"{res.original_resolution[0]}x{res.original_resolution[1]}",
                "enh_res": f"{res.enhanced_resolution[0]}x{res.enhanced_resolution[1]}",
                "orig_fps": res.original_fps,
                "enh_fps": res.enhanced_fps,
                "faces_restored": res.faces_restored,
                "elapsed": round(elapsed, 1),
            }
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


# --- TAB 3: REAL <-> ANIMATION STUDIO ---
class AnimatorPreviewRequest(BaseModel):
    input: str
    direction: str = "real_to_anim"
    style: str = "face_paint"
    strength: float = 1.0
    line_weight: float = 1.0
    color_saturation: float = 1.15
    edge_soften: float = 0.8
    film_grain: float = 0.15
    face_realism: bool = True
    face_fidelity: float = 0.85
    timestamp: float = 0.0


@app.post("/api/animator/preview")
def api_animator_preview(req: AnimatorPreviewRequest):
    try:
        inp = validate_path(req.input)
        config = AnimationConfig(
            input_path=inp,
            direction=req.direction,
            style=req.style,
            strength=req.strength,
            line_weight=req.line_weight,
            color_saturation=req.color_saturation,
            edge_soften=req.edge_soften,
            film_grain=req.film_grain,
            face_realism=req.face_realism,
            face_fidelity=req.face_fidelity,
        )
        engine = AnimatorEngine(config)
        orig_bgr, proc_bgr = engine.preview_frame(timestamp=req.timestamp)
        return {
            "success": True,
            "orig": bgr_to_base64_jpeg(orig_bgr),
            "proc": bgr_to_base64_jpeg(proc_bgr),
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


class AnimatorExecuteRequest(BaseModel):
    input: str
    output: str
    direction: str = "real_to_anim"
    style: str = "face_paint"
    strength: float = 1.0
    line_weight: float = 1.0
    color_saturation: float = 1.15
    edge_soften: float = 0.8
    film_grain: float = 0.15
    face_realism: bool = True
    face_fidelity: float = 0.85
    video_codec: str = "h264_nvenc"


@app.post("/api/animator/execute")
def api_animator_execute(req: AnimatorExecuteRequest):
    try:
        inp = validate_path(req.input)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not req.output:
        return JSONResponse(status_code=400, content={"error": "Output destination is required"})

    job_id = str(uuid.uuid4())
    job = JobState(job_id, "animator")
    JOBS[job_id] = job

    def worker():
        try:
            out_p = Path(req.output).resolve()
            config = AnimationConfig(
                input_path=inp,
                output_path=out_p,
                direction=req.direction,
                style=req.style,
                strength=req.strength,
                line_weight=req.line_weight,
                color_saturation=req.color_saturation,
                edge_soften=req.edge_soften,
                film_grain=req.film_grain,
                face_realism=req.face_realism,
                face_fidelity=req.face_fidelity,
                video_codec=req.video_codec,
            )
            engine = AnimatorEngine(config)

            def anim_callback(stage_name: str, pct: float):
                job.progress = round(pct, 1)
                job.stage = f"{stage_name} ({pct:.0f}%)"

            start_t = time.time()
            res = engine.execute(stage_callback=anim_callback)
            elapsed = time.time() - start_t

            job.progress = 100.0
            job.stage = "Stylization complete!"
            job.status = "completed"
            job.result = {
                "input": str(res.input_path),
                "output": str(res.output_path),
                "is_video": res.is_video,
                "orig_res": f"{res.original_resolution[0]}x{res.original_resolution[1]}",
                "proc_res": f"{res.processed_resolution[0]}x{res.processed_resolution[1]}",
                "duration": res.duration,
                "orig_fps": res.original_fps,
                "direction": req.direction,
                "style": req.style,
                "elapsed": round(elapsed, 1),
            }
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


# --- TAB 4: FACE STUDIO & REPLACEMENT ---
class FaceAnalyzeRequest(BaseModel):
    input: str
    sample_interval: float = 0.5
    similarity_threshold: float = 0.40


@app.post("/api/face_studio/analyze")
def api_face_analyze(req: FaceAnalyzeRequest):
    try:
        inp = validate_path(req.input)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    job_id = str(uuid.uuid4())
    job = JobState(job_id, "face_analyze")
    JOBS[job_id] = job

    def worker():
        try:
            config = FaceStudioConfig(input_path=inp, similarity_threshold=req.similarity_threshold)
            engine = FaceStudioEngine(config)

            def scan_cb(desc: str, pct: float):
                job.progress = round(pct, 1)
                job.stage = f"{desc} ({pct:.0f}%)"

            clusters = engine.analyze_faces(sample_interval=req.sample_interval, progress_cb=scan_cb)
            FACE_CLUSTERS_CACHE[job_id] = clusters

            job.progress = 100.0
            job.stage = f"Scan complete: Found {len(clusters)} distinct individuals!"
            job.status = "completed"
            job.result = {
                "clusters": [
                    {
                        "person_id": c.person_id,
                        "label": c.label,
                        "thumbnail": bgr_to_base64_jpeg(c.thumbnail_bgr),
                        "appearance_count": c.appearance_count,
                        "first_seen_sec": round(c.first_seen_sec, 2),
                        "best_score": round(c.best_score, 2),
                    }
                    for c in clusters
                ]
            }
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


class FacePreviewRequest(BaseModel):
    input: str
    analyze_job_id: Optional[str] = None
    selected_person: str
    replacement_image: str
    blend_strength: float = 1.0
    enable_gfpgan: bool = True
    gfpgan_fidelity: float = 0.70
    similarity_threshold: float = 0.40
    timestamp: float = 0.0


@app.post("/api/face_studio/preview")
def api_face_preview(req: FacePreviewRequest):
    try:
        inp = validate_path(req.input)
        rep_img_path = validate_path(req.replacement_image)

        # Retrieve cached clusters or scan frame
        clusters = FACE_CLUSTERS_CACHE.get(req.analyze_job_id or "")
        if not clusters:
            # Fallback: analyze input quickly
            config_quick = FaceStudioConfig(input_path=inp, similarity_threshold=req.similarity_threshold)
            engine_quick = FaceStudioEngine(config_quick)
            clusters = engine_quick.analyze_faces(sample_interval=2.0)

        cfg = FaceStudioConfig(
            input_path=inp,
            replacement_map={req.selected_person: rep_img_path},
            blend_strength=req.blend_strength,
            enable_gfpgan=req.enable_gfpgan,
            gfpgan_fidelity=req.gfpgan_fidelity,
            similarity_threshold=req.similarity_threshold,
        )
        engine = FaceStudioEngine(cfg)
        orig_bgr, proc_bgr = engine.preview_swap_frame(clusters, timestamp=req.timestamp)

        return {
            "success": True,
            "orig": bgr_to_base64_jpeg(orig_bgr),
            "proc": bgr_to_base64_jpeg(proc_bgr),
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


class FaceExecuteRequest(BaseModel):
    input: str
    output: str
    replacement_map: Dict[str, str]
    blend_strength: float = 1.0
    enable_gfpgan: bool = True
    gfpgan_fidelity: float = 0.70
    similarity_threshold: float = 0.40
    video_codec: str = "h264_nvenc"


@app.post("/api/face_studio/execute")
def api_face_execute(req: FaceExecuteRequest):
    try:
        inp = validate_path(req.input)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not req.output:
        return JSONResponse(status_code=400, content={"error": "Output destination is required"})
    if not req.replacement_map:
        return JSONResponse(status_code=400, content={"error": "No replacement pictures provided"})

    job_id = str(uuid.uuid4())
    job = JobState(job_id, "face_execute")
    JOBS[job_id] = job

    def worker():
        try:
            out_p = Path(req.output).resolve()
            rep_paths = {k: Path(v).resolve() for k, v in req.replacement_map.items() if Path(v).is_file()}

            config = FaceStudioConfig(
                input_path=inp,
                output_path=out_p,
                replacement_map=rep_paths,
                blend_strength=req.blend_strength,
                enable_gfpgan=req.enable_gfpgan,
                gfpgan_fidelity=req.gfpgan_fidelity,
                similarity_threshold=req.similarity_threshold,
                video_codec=req.video_codec,
            )
            engine = FaceStudioEngine(config)

            def exec_cb(desc: str, pct: float):
                job.progress = round(pct, 1)
                job.stage = f"{desc} ({pct:.0f}%)"

            start_t = time.time()
            res = engine.execute(progress_cb=exec_cb)
            elapsed = time.time() - start_t

            job.progress = 100.0
            job.stage = "Face replacement complete!"
            job.status = "completed"
            job.result = {
                "input": str(res.input_path),
                "output": str(res.output_path),
                "is_video": res.is_video,
                "resolution": f"{res.resolution[0]}x{res.resolution[1]}",
                "duration": res.duration,
                "fps": res.fps,
                "swapped_count": res.swapped_faces_count,
                "elapsed": round(elapsed, 1),
            }
        except Exception as e:
            job.status = "error"
            job.error = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


# Mount frontend static files if they exist
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
