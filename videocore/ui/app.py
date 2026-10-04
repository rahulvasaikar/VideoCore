"""VideoCore Studio - Modern Interactive Application.

Web-based graphical interface for VideoCore video joining,
smart format normalization, and GPU-accelerated AI media enhancement.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import streamlit as st

from videocore.core.ffmpeg import (
    detect_available_gpu_encoders,
    get_ffmpeg_path,
    get_ffprobe_path,
)
from videocore.core.probe import are_streams_compatible, probe_video
from videocore.tools.animator.engine import (
    AnimationConfig,
    AnimationResult,
    AnimatorEngine,
)
from videocore.tools.animator.models import (
    ANIMEGAN_MODELS,
    ensure_all_anime_models,
    get_animegan_model_path,
)
from videocore.tools.face_studio.analyzer import DetectedFace
from videocore.tools.face_studio.engine import (
    FaceStudioConfig,
    FaceStudioEngine,
    FaceStudioResult,
)
from videocore.tools.face_studio.models import ensure_face_models, get_sface_model
from videocore.tools.enhancer.engine import (
    EnhanceConfig,
    EnhancerEngine,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from videocore.tools.enhancer.models import (
    ensure_all_models,
    get_gfpgan_model,
    get_realesrgan_bin,
    get_rife_bin,
    get_yunet_model,
)
from videocore.tools.joiner.cli import (
    RESOLUTION_PRESETS,
    discover_folder_videos,
    natural_sort_key,
    parse_resolution,
)
from videocore.tools.joiner.engine import JoinConfig, JoinMode, VideoJoiner
from videocore.ui.dialogs import (
    open_in_file_explorer,
    play_video,
    select_folder,
    select_media_file,
    select_save_file,
    select_video_files,
)


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="VideoCore Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-green { background-color: #065f46; color: #6ee7b7; }
    .badge-blue { background-color: #1e3a8a; color: #93c5fd; }
    .badge-purple { background-color: #581c87; color: #d8b4fe; }
    .badge-gray { background-color: #334155; color: #cbd5e1; }
    .card {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Session State Initialization ---
def init_state():
    if "join_inputs" not in st.session_state:
        st.session_state.join_inputs = []
    if "join_folder" not in st.session_state:
        st.session_state.join_folder = ""
    if "join_output_path" not in st.session_state:
        st.session_state.join_output_path = ""
    if "last_join_result" not in st.session_state:
        st.session_state.last_join_result = None

    if "join_format" not in st.session_state:
        st.session_state.join_format = ".mp4"

    if "enhance_input" not in st.session_state:
        st.session_state.enhance_input = ""
    if "enhance_output" not in st.session_state:
        st.session_state.enhance_output = ""
    if "last_enhance_result" not in st.session_state:
        st.session_state.last_enhance_result = None
    if "enhance_video_format" not in st.session_state:
        st.session_state.enhance_video_format = ".mp4"
    if "enhance_image_format" not in st.session_state:
        st.session_state.enhance_image_format = ".png"

    if "anim_input" not in st.session_state:
        st.session_state.anim_input = ""
    if "anim_output" not in st.session_state:
        st.session_state.anim_output = ""
    if "last_anim_result" not in st.session_state:
        st.session_state.last_anim_result = None
    if "anim_direction" not in st.session_state:
        st.session_state.anim_direction = "real_to_anim"
    if "anim_video_format" not in st.session_state:
        st.session_state.anim_video_format = ".mp4"
    if "anim_image_format" not in st.session_state:
        st.session_state.anim_image_format = ".png"
    if "anim_preview_orig" not in st.session_state:
        st.session_state.anim_preview_orig = None
    if "anim_preview_proc" not in st.session_state:
        st.session_state.anim_preview_proc = None

    if "face_studio_input" not in st.session_state:
        st.session_state.face_studio_input = ""
    if "face_studio_output" not in st.session_state:
        st.session_state.face_studio_output = ""
    if "face_detected_clusters" not in st.session_state:
        st.session_state.face_detected_clusters = []
    if "face_replacement_map" not in st.session_state:
        st.session_state.face_replacement_map = {}
    if "face_preview_orig" not in st.session_state:
        st.session_state.face_preview_orig = None
    if "face_preview_proc" not in st.session_state:
        st.session_state.face_preview_proc = None
    if "last_face_studio_result" not in st.session_state:
        st.session_state.last_face_studio_result = None
    if "face_studio_video_format" not in st.session_state:
        st.session_state.face_studio_video_format = ".mp4"
    if "face_studio_image_format" not in st.session_state:
        st.session_state.face_studio_image_format = ".png"

init_state()



# --- Sidebar: System Status & Hardware ---
with st.sidebar:
    st.markdown("### ⚙️ System & Hardware")
    
    # GPU status check
    gpu_map = detect_available_gpu_encoders()
    has_nvenc = gpu_map.get("nvenc", False)
    
    if has_nvenc:
        st.markdown('<span class="stat-badge badge-green">🚀 NVIDIA NVENC Active</span>', unsafe_allow_html=True)
        st.caption("GPU acceleration available for ultra-fast encoding.")
    else:
        st.markdown('<span class="stat-badge badge-gray">🖥️ CPU Encoding Mode</span>', unsafe_allow_html=True)
        st.caption("Hardware acceleration not detected; using multi-threaded CPU.")

    st.divider()

    # Binaries & Tooling
    st.markdown("##### Core Binaries")
    ffmpeg_bin = get_ffmpeg_path()
    ffprobe_bin = get_ffprobe_path()
    
    st.text(f"FFmpeg: {Path(ffmpeg_bin).name}")
    st.text(f"FFprobe: {Path(ffprobe_bin).name if ffprobe_bin else 'Missing'}")

    st.markdown("##### AI Models")
    esrgan_ok = get_realesrgan_bin() is not None
    rife_ok = get_rife_bin() is not None
    yunet_ok = get_yunet_model() is not None
    gfpgan_ok = get_gfpgan_model() is not None
    hayao_ok = get_animegan_model_path("hayao") is not None
    shinkai_ok = get_animegan_model_path("shinkai") is not None
    paprika_ok = get_animegan_model_path("paprika") is not None

    sface_ok = get_sface_model() is not None

    cols_ai = st.columns(2)
    with cols_ai[0]:
        st.markdown(f"{'✅' if esrgan_ok else '❌'} Real-ESRGAN")
        st.markdown(f"{'✅' if rife_ok else '❌'} RIFE 60fps")
        st.markdown(f"{'✅' if hayao_ok else '❌'} Anime (Hayao)")
    with cols_ai[1]:
        st.markdown(f"{'✅' if yunet_ok else '❌'} YuNet Face")
        st.markdown(f"{'✅' if sface_ok else '❌'} SFace Match")
        st.markdown(f"{'✅' if gfpgan_ok else '❌'} GFPGAN v1.4")

    all_models_present = (
        esrgan_ok and rife_ok and yunet_ok and sface_ok and gfpgan_ok and hayao_ok and shinkai_ok and paprika_ok
    )
    if not all_models_present:
        if st.button("📥 Download Missing Models"):
            with st.spinner("Downloading AI models and binaries..."):
                ensure_all_models()
                ensure_all_anime_models()
                ensure_face_models()
            st.success("All models downloaded!")
            st.rerun()

    st.divider()
    st.caption("VideoCore v0.1.0 • High-Performance Media Suite")


# --- Main Header ---
st.markdown('<div class="main-header">VideoCore Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Lossless Video Concatenation, AI Enhancement, Stylization & Face Studio</div>',
    unsafe_allow_html=True,
)


# --- Tabs ---
tab_joiner, tab_enhancer, tab_animator, tab_face_studio, tab_diagnostics = st.tabs([
    "🎬 Video Joiner",
    "✨ AI Media Enhancer",
    "🎭 Real ⇄ Animation Studio",
    "👥 Face Studio & Replacement",
    "📊 Video Inspector & Specs",
])




# ==============================================================================
# TAB 1: VIDEO JOINER
# ==============================================================================
with tab_joiner:
    st.markdown("#### 1. Input Videos Selection")
    
    col_pick_folder, col_pick_files, col_clear = st.columns([1.2, 1.2, 0.8])
    with col_pick_folder:
        if st.button("📁 Browse Video Folder...", width="stretch"):
            chosen = select_folder(title="Select Folder Containing Videos to Join")
            if chosen:
                st.session_state.join_folder = chosen
                videos = discover_folder_videos(Path(chosen))
                st.session_state.join_inputs = [str(v.resolve()) for v in videos]
                if videos and not st.session_state.join_output_path:
                    folder_name = Path(chosen).name.strip()
                    ext = st.session_state.join_format
                    st.session_state.join_output_path = str((Path(chosen) / f"{folder_name}_joined{ext}").resolve())
                st.rerun()

    with col_pick_files:
        if st.button("📄 Select Video Files...", width="stretch"):
            chosen_files = select_video_files(title="Select Video Files to Join")
            if chosen_files:
                st.session_state.join_inputs = chosen_files
                st.session_state.join_folder = ""
                if not st.session_state.join_output_path:
                    first_p = Path(chosen_files[0])
                    ext = st.session_state.join_format
                    st.session_state.join_output_path = str((first_p.parent / f"joined_output{ext}").resolve())
                st.rerun()

    with col_clear:
        if st.button("🧹 Clear Selection", width="stretch"):
            st.session_state.join_inputs = []
            st.session_state.join_folder = ""
            st.session_state.last_join_result = None
            st.rerun()

    # Direct Path Inputs (Allows quick copy/paste)
    folder_text = st.text_input(
        "Or paste folder path directly:",
        value=st.session_state.join_folder,
        placeholder="G:\\Videos\\TripClips",
    )
    if folder_text and folder_text != st.session_state.join_folder:
        p = Path(folder_text)
        if p.is_dir():
            st.session_state.join_folder = folder_text
            videos = discover_folder_videos(p)
            st.session_state.join_inputs = [str(v.resolve()) for v in videos]
            if videos and not st.session_state.join_output_path:
                ext = st.session_state.join_format
                st.session_state.join_output_path = str((p / f"{p.name}_joined{ext}").resolve())
            st.rerun()

    # Display video list & Probe Metadata
    current_files = [Path(p) for p in st.session_state.join_inputs if Path(p).is_file()]

    if not current_files:
        st.info("👋 Select a folder or video files above to begin sequence arrangement.")
    else:
        st.markdown(f"##### Selected Video Sequence ({len(current_files)} clips)")
        
        # Action row for reordering
        col_sort_nat, col_sort_rev, col_sort_asc, _ = st.columns([1, 1, 1, 3])
        with col_sort_nat:
            if st.button("🔤 Natural Sort", width="stretch"):
                current_files.sort(key=lambda p: natural_sort_key(p.name))
                st.session_state.join_inputs = [str(p.resolve()) for p in current_files]
                st.rerun()
        with col_sort_rev:
            if st.button("🔄 Reverse Order", width="stretch"):
                current_files.reverse()
                st.session_state.join_inputs = [str(p.resolve()) for p in current_files]
                st.rerun()
        with col_sort_asc:
            if st.button("🔠 A-Z Sort", width="stretch"):
                current_files.sort(key=lambda p: p.name.lower())
                st.session_state.join_inputs = [str(p.resolve()) for p in current_files]
                st.rerun()

        # Probing and table display
        metas = []
        total_duration = 0.0
        table_rows = []

        with st.spinner("Analyzing stream properties..."):
            for idx, p in enumerate(current_files, start=1):
                try:
                    meta = probe_video(p)
                    metas.append(meta)
                    total_duration += meta.duration
                    pv = meta.primary_video
                    pa = meta.primary_audio
                    table_rows.append({
                        "#": idx,
                        "Filename": p.name,
                        "Resolution": f"{pv.width}x{pv.height}" if pv else "N/A",
                        "FPS": f"{pv.fps:.2f}" if pv else "N/A",
                        "Duration": f"{meta.duration:.1f}s",
                        "Codec": pv.codec_name if pv else "N/A",
                        "Audio": f"{pa.codec_name} ({pa.sample_rate}Hz)" if pa else "None",
                    })
                except Exception as e:
                    table_rows.append({
                        "#": idx,
                        "Filename": p.name,
                        "Resolution": "Error",
                        "FPS": "Error",
                        "Duration": "N/A",
                        "Codec": str(e),
                        "Audio": "N/A",
                    })

        st.dataframe(table_rows, width="stretch", hide_index=True)

        # Sequence Reorder / Remove Controls
        with st.expander("🛠️ Reorder or Remove Individual Clips"):
            for i, p in enumerate(current_files):
                c_name, c_up, c_down, c_del = st.columns([4, 1, 1, 1])
                with c_name:
                    st.text(f"{i + 1}. {p.name}")
                with c_up:
                    if st.button("⬆️", key=f"up_{i}", disabled=(i == 0)):
                        current_files[i - 1], current_files[i] = current_files[i], current_files[i - 1]
                        st.session_state.join_inputs = [str(x.resolve()) for x in current_files]
                        st.rerun()
                with c_down:
                    if st.button("⬇️", key=f"down_{i}", disabled=(i == len(current_files) - 1)):
                        current_files[i + 1], current_files[i] = current_files[i], current_files[i + 1]
                        st.session_state.join_inputs = [str(x.resolve()) for x in current_files]
                        st.rerun()
                with c_del:
                    if st.button("❌", key=f"del_{i}"):
                        current_files.pop(i)
                        st.session_state.join_inputs = [str(x.resolve()) for x in current_files]
                        st.rerun()

        # Compatibility & Mode Assessment
        if metas and len(metas) == len(current_files):
            compat, reason = are_streams_compatible(metas)
            c_mode1, c_mode2 = st.columns([1, 2])
            with c_mode1:
                st.metric("Total Duration", f"{total_duration:.1f} s", f"{len(current_files)} clips")
            with c_mode2:
                if compat:
                    st.success(f"⚡ Fast Lossless Copy Compatible: {reason}")
                else:
                    st.warning(f"📐 Normalization Needed: {reason}")

        st.divider()

        # Output & Encoding Configuration
        st.markdown("#### 2. Join & Encoding Settings")
        
        col_out_path, col_out_fmt, col_out_btn = st.columns([2.6, 1.1, 1.3])
        with col_out_fmt:
            video_formats = [".mp4", ".mkv", ".mov", ".avi", ".webm"]
            curr_fmt = Path(st.session_state.join_output_path).suffix.lower() if st.session_state.join_output_path else st.session_state.join_format
            if curr_fmt not in video_formats:
                curr_fmt = ".mp4"
            fmt_idx = video_formats.index(curr_fmt)
            selected_fmt = st.selectbox(
                "Output Format",
                options=video_formats,
                index=fmt_idx,
                format_func=lambda x: f"{x.upper()[1:]} ({x})",
                key="join_format_dropdown",
            )
            if selected_fmt != curr_fmt:
                st.session_state.join_format = selected_fmt
                if st.session_state.join_output_path:
                    st.session_state.join_output_path = str(Path(st.session_state.join_output_path).with_suffix(selected_fmt))
                st.rerun()

        with col_out_path:
            output_str = st.text_input(
                "Output Video Destination:",
                value=st.session_state.join_output_path,
                placeholder=f"G:\\Videos\\combined{selected_fmt}",
            )
            st.session_state.join_output_path = output_str

        with col_out_btn:
            st.write("")
            st.write("")
            if st.button("💾 Browse Destination...", width="stretch"):
                def_name = Path(output_str).name if output_str else f"joined_output{selected_fmt}"
                save_chosen = select_save_file(
                    title="Choose Output Video Location",
                    default_name=def_name,
                    initial_dir=str(Path(output_str).parent) if output_str else None,
                )
                if save_chosen:
                    st.session_state.join_output_path = save_chosen
                    st.rerun()

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            join_mode = st.selectbox(
                "Join Mode",
                options=["auto", "fast", "normalize"],
                format_func=lambda x: {
                    "auto": "Auto (Fastest Safe Choice)",
                    "fast": "Fast (Lossless Copy -c copy)",
                    "normalize": "Normalize (Re-encode canvas)",
                }[x],
                help="Auto will use lossless instant stream copy if clips match, or re-encode if resolutions differ.",
            )

            res_options = ["Match First Clip", "1080p (1920x1080)", "720p (1280x720)", "4K (3840x2160)", "Vertical 1080p (1080x1920)", "Vertical 720p (720x1280)", "Custom"]
            res_choice = st.selectbox("Resolution Preset", options=res_options)
            custom_res = None
            if res_choice == "Custom":
                custom_res_str = st.text_input("Custom Width x Height", value="1920x1080")
                try:
                    custom_res = parse_resolution(custom_res_str)
                except Exception:
                    custom_res = None
            elif res_choice != "Match First Clip":
                res_lower = res_choice.lower()
                if "vertical" in res_lower and "1080p" in res_lower:
                    key = "vertical_1080p"
                elif "vertical" in res_lower and "720p" in res_lower:
                    key = "vertical_720p"
                else:
                    key = res_choice.split(" ")[0].lower()
                custom_res = RESOLUTION_PRESETS.get(key)

        with col_m2:
            fps_choice = st.selectbox(
                "Target Framerate",
                options=["Match First Clip", "60 FPS", "30 FPS", "24 FPS"],
            )
            target_fps = None
            if fps_choice == "60 FPS":
                target_fps = 60.0
            elif fps_choice == "30 FPS":
                target_fps = 30.0
            elif fps_choice == "24 FPS":
                target_fps = 24.0

            gpu_options = ["auto", "nvenc", "qsv", "amf", "cpu"]
            selected_gpu = st.selectbox(
                "Hardware Acceleration",
                options=gpu_options,
                format_func=lambda x: {
                    "auto": "Auto (NVIDIA NVENC if available)",
                    "nvenc": "NVIDIA NVENC (Fast GPU)",
                    "qsv": "Intel QuickSync (QSV)",
                    "amf": "AMD AMF",
                    "cpu": "CPU Only (libx264)",
                }[x],
            )

        with col_m3:
            crf_val = st.slider("Quality (CRF)", min_value=15, max_value=35, value=20, help="Lower = higher quality. 18-22 is visually lossless.")
            preset_val = st.selectbox("Preset Speed", options=["fast", "medium", "slow", "faster", "veryslow"], index=1)

        st.divider()

        # Start Button
        btn_join = st.button("🚀 Start Joining Videos", type="primary", width="stretch")

        if btn_join:
            if not output_str:
                st.error("Please specify an output destination.")
            else:
                out_path = Path(output_str).resolve()
                config = JoinConfig(
                    inputs=current_files,
                    output=out_path,
                    mode=JoinMode(join_mode),
                    target_resolution=custom_res,
                    target_fps=target_fps,
                    crf=crf_val,
                    preset=preset_val,
                    gpu=selected_gpu,
                    overwrite=True,
                )

                joiner = VideoJoiner(config)
                
                # Progress display elements
                progress_bar = st.progress(0, text="Initializing FFmpeg engine...")
                status_box = st.empty()
                
                def on_progress(fraction: float, speed: str, fps: str, time_str: str):
                    pct = int(fraction * 100)
                    progress_bar.progress(min(100, max(0, pct)), text=f"Joining: {pct}% complete")
                    status_box.caption(f"⚡ Time: **{time_str}** | Speed: **{speed}** | FPS: **{fps}**")

                def on_status(msg: str):
                    status_box.info(msg)

                try:
                    start_time = time.time()
                    result = joiner.join(on_progress=on_progress, on_status=on_status)
                    elapsed = time.time() - start_time
                    progress_bar.progress(100, text="Done!")
                    
                    st.session_state.last_join_result = {
                        "path": str(result.output_path),
                        "duration": result.total_duration,
                        "size_mb": result.output_size_bytes / (1024 * 1024),
                        "mode": result.mode_used.value.upper(),
                        "encoder": result.encoder_used,
                        "elapsed": elapsed,
                    }
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Join operation failed: {e}")

    # Display Last Result & Player
    if st.session_state.last_join_result:
        res = st.session_state.last_join_result
        st.markdown("---")
        st.markdown("### 🎉 Output Ready")
        
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        c_res1.metric("File Size", f"{res['size_mb']:.2f} MB")
        c_res2.metric("Duration", f"{res['duration']:.1f} s")
        c_res3.metric("Processing Time", f"{res['elapsed']:.1f} s")
        c_res4.metric("Mode / Encoder", res['mode'], res['encoder'])

        col_play, col_open, _ = st.columns([1, 1, 2])
        with col_play:
            if st.button("▶️ Play Video", width="stretch"):
                play_video(res["path"])
        with col_open:
            if st.button("📂 Open in Windows Explorer", width="stretch"):
                open_in_file_explorer(res["path"])

        if Path(res["path"]).is_file():
            if res.get("size_mb", 0) > 100:
                st.info(f"💡 Large output video ({res['size_mb']:.1f} MB). Click **▶️ Play Video** to play instantly in your default media player without browser buffering.")
                with st.expander("🌐 Preview in browser anyway"):
                    try:
                        st.video(res["path"])
                    except Exception as e:
                        st.warning(f"Browser player could not load large video: {e}")
            else:
                try:
                    st.video(res["path"])
                except Exception as e:
                    st.warning(f"Browser player could not load video: {e}")


# ==============================================================================
# TAB 2: AI MEDIA ENHANCER
# ==============================================================================
with tab_enhancer:
    st.markdown("#### 1. Select Media (Image or Video)")
    
    col_en_pick, col_en_clear = st.columns([1.5, 0.5])
    with col_en_pick:
        if st.button("🖼️ Browse Image or Video File...", width="stretch"):
            chosen = select_media_file(title="Select Media to Enhance")
            if chosen:
                st.session_state.enhance_input = chosen
                inp_p = Path(chosen)
                chosen_ext = inp_p.suffix.lower()
                target_ext = st.session_state.enhance_video_format if chosen_ext in VIDEO_EXTENSIONS else st.session_state.enhance_image_format
                st.session_state.enhance_output = str(inp_p.parent / f"{inp_p.stem}_enhanced{target_ext}")
                st.rerun()
    with col_en_clear:
        if st.button("🧹 Clear", key="clear_enhancer", width="stretch"):
            st.session_state.enhance_input = ""
            st.session_state.enhance_output = ""
            st.session_state.last_enhance_result = None
            st.rerun()

    input_media_str = st.text_input(
        "Or paste media path directly:",
        value=st.session_state.enhance_input,
        placeholder="G:\\Photos\\portrait.jpg or G:\\Videos\\clip.mp4",
    )
    if input_media_str and input_media_str != st.session_state.enhance_input:
        st.session_state.enhance_input = input_media_str
        inp_p = Path(input_media_str)
        if inp_p.is_file():
            chosen_ext = inp_p.suffix.lower()
            target_ext = st.session_state.enhance_video_format if chosen_ext in VIDEO_EXTENSIONS else st.session_state.enhance_image_format
            st.session_state.enhance_output = str(inp_p.parent / f"{inp_p.stem}_enhanced{target_ext}")
            st.rerun()

    current_media = Path(st.session_state.enhance_input)
    if not current_media.is_file():
        st.info("👋 Select an image (JPG, PNG, WEBP) or video (MP4, MKV, MOV) to enhance.")
    else:
        ext = current_media.suffix.lower()
        is_image = ext in IMAGE_EXTENSIONS
        is_video = ext in VIDEO_EXTENSIONS

        # Preview input
        col_preview, col_meta = st.columns([1, 1])
        with col_preview:
            st.markdown("##### Input Preview")
            if is_image:
                st.image(str(current_media), width="stretch")
            elif is_video:
                try:
                    if current_media.stat().st_size > 100 * 1024 * 1024:
                        st.caption(f"Large preview file ({current_media.stat().st_size / (1024*1024):.1f} MB)")
                    else:
                        st.video(str(current_media))
                except Exception as e:
                    st.caption(f"Input video preview unavailable: {e}")
        with col_meta:
            st.markdown("##### Media Properties")
            try:
                meta = probe_video(current_media)
                pv = meta.primary_video
                st.text(f"File: {current_media.name}")
                st.text(f"Type: {'Image' if is_image else 'Video'}")
                if pv:
                    st.text(f"Resolution: {pv.width} x {pv.height}")
                    st.text(f"FPS: {pv.fps:.2f}")
                    st.text(f"Codec: {pv.codec_name}")
                if is_video:
                    total_frames = int(meta.duration * (pv.fps if pv else 30.0))
                    duration_min = meta.duration / 60.0
                    st.text(f"Duration: {duration_min:.1f} min ({meta.duration:.1f} s)")
                    st.text(f"Total Frames: ~{total_frames:,}")
                    if meta.duration > 180:
                        st.caption("ℹ️ Long video detected. Temporary frames will be processed on the drive with the most free space (e.g. Drive H:).")
            except Exception as e:
                st.caption(f"Could not probe: {e}")

        st.divider()

        # Enhancement Controls
        st.markdown("#### 2. Enhancement Options")

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.markdown("##### 🔍 Super-Resolution Upscaling")
            scale_factor = st.radio(
                "Upscale Multiplier",
                options=[1, 2, 4],
                index=2 if is_image else 1,
                format_func=lambda x: f"{x}x Upscale" if x > 1 else "1x (No Upscale)",
            )
            model_name = st.selectbox(
                "Upscaling Model",
                options=["realesrgan-x4plus", "realesrgan-x4plus-anime"],
                format_func=lambda x: "Real-ESRGAN x4 Plus (Photos/Real World)" if "anime" not in x else "Real-ESRGAN Anime (Cartoons/Art)",
            )

        with col_e2:
            st.markdown("##### 👤 Facial Restoration (GFPGAN)")
            face_restore = st.checkbox("Enable AI Face Restoration", value=True, help="Reconstructs eyes, mouth, and skin texture using GFPGAN v1.4.")
            face_fidelity = st.slider("Face Fidelity", min_value=0.1, max_value=1.0, value=0.7, step=0.05, help="Higher keeps more original person identity.")

        with col_e3:
            st.markdown("##### 🚀 Motion Smoothing & Codec")
            if is_video:
                interpolate = st.checkbox("Smooth Motion to 60 FPS (RIFE)", value=True)
                target_fps = 60.0 if interpolate else None
                video_codec = st.selectbox(
                    "Output Video Codec",
                    options=["h264_nvenc", "hevc_nvenc", "libx264"],
                    format_func=lambda x: {
                        "h264_nvenc": "H.264 (NVIDIA NVENC Fast)",
                        "hevc_nvenc": "HEVC/H.265 (NVIDIA NVENC)",
                        "libx264": "H.264 (CPU libx264)",
                    }[x],
                )
            else:
                st.caption("Motion smoothing is only applicable to video files.")
                interpolate = False
                target_fps = None
                video_codec = "h264_nvenc"

        # Output Destination
        out_col1, out_col_fmt, out_col2 = st.columns([2.6, 1.1, 1.3])
        with out_col_fmt:
            if is_video:
                formats = [".mp4", ".mkv", ".mov", ".avi"]
                curr_fmt = Path(st.session_state.enhance_output).suffix.lower() if st.session_state.enhance_output else st.session_state.enhance_video_format
                if curr_fmt not in formats:
                    curr_fmt = ".mp4"
                f_idx = formats.index(curr_fmt)
                chosen_fmt = st.selectbox(
                    "Output Format",
                    options=formats,
                    index=f_idx,
                    format_func=lambda x: f"{x.upper()[1:]} ({x})",
                    key="enhance_video_fmt_dropdown",
                )
                if chosen_fmt != curr_fmt:
                    st.session_state.enhance_video_format = chosen_fmt
                    if st.session_state.enhance_output:
                        st.session_state.enhance_output = str(Path(st.session_state.enhance_output).with_suffix(chosen_fmt))
                    st.rerun()
            else:
                formats = [".png", ".jpg", ".webp"]
                curr_fmt = Path(st.session_state.enhance_output).suffix.lower() if st.session_state.enhance_output else st.session_state.enhance_image_format
                if curr_fmt not in formats:
                    curr_fmt = ".png"
                f_idx = formats.index(curr_fmt)
                chosen_fmt = st.selectbox(
                    "Output Format",
                    options=formats,
                    index=f_idx,
                    format_func=lambda x: f"{x.upper()[1:]} ({x})",
                    key="enhance_image_fmt_dropdown",
                )
                if chosen_fmt != curr_fmt:
                    st.session_state.enhance_image_format = chosen_fmt
                    if st.session_state.enhance_output:
                        st.session_state.enhance_output = str(Path(st.session_state.enhance_output).with_suffix(chosen_fmt))
                    st.rerun()

        with out_col1:
            en_out_str = st.text_input("Output Destination:", value=st.session_state.enhance_output)
            st.session_state.enhance_output = en_out_str

        with out_col2:
            st.write("")
            st.write("")
            if st.button("💾 Save As...", width="stretch"):
                def_name = Path(en_out_str).name if en_out_str else f"{current_media.stem}_enhanced{chosen_fmt}"
                save_chosen = select_save_file(
                    title="Choose Enhanced File Destination",
                    default_name=def_name,
                    initial_dir=str(Path(en_out_str).parent) if en_out_str else None,
                )
                if save_chosen:
                    st.session_state.enhance_output = save_chosen
                    st.rerun()

        btn_enhance = st.button("✨ Run AI Enhancement", type="primary", width="stretch")

        if btn_enhance:
            if not en_out_str:
                st.error("Please specify an output destination.")
            else:
                out_path = Path(en_out_str).resolve()
                enhance_config = EnhanceConfig(
                    input_path=current_media,
                    output_path=out_path,
                    scale=scale_factor,
                    model_name=model_name,
                    face_restore=face_restore,
                    face_fidelity=face_fidelity,
                    interpolate=interpolate,
                    target_fps=target_fps,
                    tile_size=400,
                    gpu_id=0,
                    video_codec=video_codec,
                )

                engine = EnhancerEngine(enhance_config)
                progress_bar = st.progress(0, text="Starting AI Pipeline...")
                status_box = st.empty()

                def stage_callback(stage_name: str, pct: float):
                    progress_bar.progress(int(pct), text=f"Stage: {stage_name} ({pct:.0f}%)")
                    status_box.info(f"🔄 **{stage_name}**")

                try:
                    start_time = time.time()
                    res = engine.execute(stage_callback=stage_callback)
                    elapsed = time.time() - start_time
                    progress_bar.progress(100, text="Complete!")
                    
                    st.session_state.last_enhance_result = {
                        "input": str(res.input_path),
                        "output": str(res.output_path),
                        "is_video": res.is_video,
                        "orig_res": f"{res.original_resolution[0]}x{res.original_resolution[1]}",
                        "enh_res": f"{res.enhanced_resolution[0]}x{res.enhanced_resolution[1]}",
                        "orig_fps": res.original_fps,
                        "enh_fps": res.enhanced_fps,
                        "faces_restored": res.faces_restored,
                        "elapsed": elapsed,
                    }
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"AI Enhancement failed: {e}")

    # Display Last Result
    if st.session_state.last_enhance_result:
        res = st.session_state.last_enhance_result
        st.markdown("---")
        st.markdown("### ✨ Enhancement Complete!")

        c_en1, c_en2, c_en3 = st.columns(3)
        c_en1.metric("Resolution", f"{res['enh_res']}", f"from {res['orig_res']}")
        if res["is_video"] and res["enh_fps"]:
            c_en2.metric("Framerate", f"{res['enh_fps']:.1f} fps", f"from {res['orig_fps']:.1f} fps")
        else:
            c_en2.metric("Faces Restored", "Yes (GFPGAN)" if res['faces_restored'] else "No")
        c_en3.metric("Processing Time", f"{res['elapsed']:.1f} s")

        col_play_en, col_open_en, _ = st.columns([1, 1, 2])
        with col_play_en:
            if res["is_video"]:
                if st.button("▶️ Play Video", key="play_en_btn", width="stretch"):
                    play_video(res["output"])
        with col_open_en:
            if st.button("📂 Show in Explorer", key="open_en_exp", width="stretch"):
                open_in_file_explorer(res["output"])

        if not res["is_video"]:
            st.markdown("##### Before vs After")
            col_b, col_a = st.columns(2)
            with col_b:
                st.caption("Original")
                st.image(res["input"], width="stretch")
            with col_a:
                st.caption("AI Enhanced")
                st.image(res["output"], width="stretch")
        else:
            st.markdown("##### Enhanced Video Player")
            out_file = Path(res["output"])
            if out_file.is_file():
                size_mb = out_file.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    st.info(f"💡 Large output video ({size_mb:.1f} MB). Click **▶️ Play Video** to play instantly in your default media player.")
                    with st.expander("🌐 Preview in browser anyway"):
                        try:
                            st.video(res["output"])
                        except Exception as e:
                            st.warning(f"Browser player could not load large video: {e}")
                else:
                    try:
                        st.video(res["output"])
                    except Exception as e:
                        st.warning(f"Browser player could not load video: {e}")


# ==============================================================================
# TAB 3: REAL <-> ANIMATION STUDIO
# ==============================================================================
with tab_animator:
    st.markdown("#### 🎭 Real ⇄ Animation Studio")
    st.caption("Bidirectional transformation: Convert real video & photos into anime / cartoons, or convert 2D animation back into photorealistic video.")

    # Direction selector
    dir_choice = st.radio(
        "Transformation Direction",
        options=["real_to_anim", "anim_to_real"],
        format_func=lambda x: "🎬 Real Video / Photo ➔ Animation (Anime & Cartoon)" if x == "real_to_anim" else "🎨 Animation / Anime ➔ Real Video (Photorealism & Film)",
        horizontal=True,
        key="anim_dir_radio",
    )
    st.session_state.anim_direction = dir_choice

    st.markdown("#### 1. Select Media (Video or Image)")
    col_an_pick, col_an_clear = st.columns([1.5, 0.5])
    with col_an_pick:
        btn_label = "🖼️ Browse Real Video / Photo..." if dir_choice == "real_to_anim" else "🎨 Browse Animation / Cartoon Media..."
        if st.button(btn_label, key="anim_browse_btn", width="stretch"):
            chosen = select_media_file(title="Select Media to Transform")
            if chosen:
                st.session_state.anim_input = chosen
                inp_p = Path(chosen)
                chosen_ext = inp_p.suffix.lower()
                target_ext = st.session_state.anim_video_format if chosen_ext in VIDEO_EXTENSIONS else st.session_state.anim_image_format
                tag = "anime" if dir_choice == "real_to_anim" else "photoreal"
                st.session_state.anim_output = str(inp_p.parent / f"{inp_p.stem}_{tag}{target_ext}")
                st.session_state.anim_preview_orig = None
                st.session_state.anim_preview_proc = None
                st.rerun()
    with col_an_clear:
        if st.button("🧹 Clear", key="clear_animator", width="stretch"):
            st.session_state.anim_input = ""
            st.session_state.anim_output = ""
            st.session_state.last_anim_result = None
            st.session_state.anim_preview_orig = None
            st.session_state.anim_preview_proc = None
            st.rerun()

    input_anim_str = st.text_input(
        "Or paste media path directly:",
        value=st.session_state.anim_input,
        placeholder="G:\\Videos\\real_clip.mp4 or G:\\Videos\\anime_clip.mp4",
        key="anim_input_text",
    )
    if input_anim_str and input_anim_str != st.session_state.anim_input:
        st.session_state.anim_input = input_anim_str
        inp_p = Path(input_anim_str)
        if inp_p.is_file():
            chosen_ext = inp_p.suffix.lower()
            target_ext = st.session_state.anim_video_format if chosen_ext in VIDEO_EXTENSIONS else st.session_state.anim_image_format
            tag = "anime" if dir_choice == "real_to_anim" else "photoreal"
            st.session_state.anim_output = str(inp_p.parent / f"{inp_p.stem}_{tag}{target_ext}")
            st.session_state.anim_preview_orig = None
            st.session_state.anim_preview_proc = None
            st.rerun()

    curr_anim_media = Path(st.session_state.anim_input)
    if not curr_anim_media.is_file():
        st.info("👋 Select an image or video file above to get started.")
    else:
        a_ext = curr_anim_media.suffix.lower()
        a_is_image = a_ext in IMAGE_EXTENSIONS
        a_is_video = a_ext in VIDEO_EXTENSIONS

        col_a_prev, col_a_meta = st.columns([1, 1])
        with col_a_prev:
            st.markdown("##### Source Preview")
            if a_is_image:
                st.image(str(curr_anim_media), width="stretch")
            elif a_is_video:
                try:
                    if curr_anim_media.stat().st_size > 100 * 1024 * 1024:
                        st.caption(f"Source video ({curr_anim_media.stat().st_size / (1024*1024):.1f} MB)")
                    else:
                        st.video(str(curr_anim_media))
                except Exception as e:
                    st.caption(f"Source video preview unavailable: {e}")
        with col_a_meta:
            st.markdown("##### Source Properties")
            anim_duration = 0.0
            try:
                a_meta = probe_video(curr_anim_media)
                a_pv = a_meta.primary_video
                st.text(f"File: {curr_anim_media.name}")
                st.text(f"Type: {'Image' if a_is_image else 'Video'}")
                if a_pv:
                    st.text(f"Resolution: {a_pv.width} x {a_pv.height}")
                    st.text(f"Framerate: {a_pv.fps:.2f} fps")
                    st.text(f"Codec: {a_pv.codec_name}")
                if a_is_video:
                    anim_duration = a_meta.duration or 0.0
                    total_f = int(anim_duration * (a_pv.fps if a_pv else 30.0))
                    st.text(f"Duration: {anim_duration:.1f} s ({anim_duration/60:.1f} min)")
                    st.text(f"Total Frames: ~{total_f:,}")
            except Exception as e:
                st.caption(f"Could not probe media: {e}")

        st.divider()

        # Step 2: Stylization Options
        st.markdown("#### 2. Transformation Settings")

        if dir_choice == "real_to_anim":
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                st.markdown("##### 🎨 Animation Style")
                style_choice = st.selectbox(
                    "Style Preset",
                    options=["hayao", "shinkai", "paprika", "cartoon", "comic", "watercolor"],
                    format_func=lambda x: {
                        "hayao": "Studio Ghibli (Hayao) [AI Neural]",
                        "shinkai": "Makoto Shinkai (Your Name) [AI Neural]",
                        "paprika": "Satoshi Kon (Paprika) [AI Neural]",
                        "cartoon": "Cel-Shaded Cartoon [Fast Filter]",
                        "comic": "Comic / Manga Ink [Fast Filter]",
                        "watercolor": "Watercolor Painterly [Fast Filter]",
                    }[x],
                    key="anim_style_select",
                )
                strength = st.slider("Effect Intensity", 0.0, 1.0, 1.0, 0.05, help="Blend between original and stylized look.")
            with c_s2:
                st.markdown("##### ✏️ Contour & Line Art")
                line_weight = st.slider("Line Art Thickness", 0.5, 2.5, 1.0, 0.1, help="Thickness and emphasis of drawn outlines.")
            with c_s3:
                st.markdown("##### 🌈 Color & Vibrance")
                color_saturation = st.slider("Color Saturation", 0.5, 2.0, 1.1, 0.05, help="Boost or soften anime color saturation.")

            # Defaults for real_to_anim
            edge_soften = 0.7
            film_grain = 0.15
            face_realism = False
            face_fidelity = 0.6

        else:
            # Animation to Real
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                st.markdown("##### 📷 Delineation & Smoothing")
                edge_soften = st.slider(
                    "Delineation Factor", 0.0, 1.0, 0.7, 0.05,
                    help="Suppresses and dissolves harsh drawn 2D cartoon outlines.",
                )
                strength = st.slider("Effect Strength", 0.0, 1.0, 1.0, 0.05)
                style_choice = "photoreal"
                line_weight = 1.0
            with c_r2:
                st.markdown("##### 🎞️ 35mm Film Grain & Sensor")
                film_grain = st.slider(
                    "Physical Film Grain", 0.0, 0.5, 0.15, 0.02,
                    help="Injects realistic camera sensor noise / film grain to replace flat digital fills.",
                )
                color_saturation = st.slider(
                    "Color Saturation Balance", 0.5, 1.5, 0.95, 0.05,
                    help="Pulls back neon cartoon saturation towards natural camera tones.",
                )
            with c_r3:
                st.markdown("##### 👤 AI Face Realism (GFPGAN)")
                face_realism = st.checkbox("Reconstruct Photorealistic Faces", value=True, help="Synthesizes real photographic skin, eyes, and facial structures.")
                face_fidelity = st.slider("Face Fidelity", 0.1, 1.0, 0.6, 0.05, help="Higher preserves more of the original character structure.")

        st.divider()

        # Step 3: Instant Frame Preview
        st.markdown("#### 3. ⚡ Instant Frame Preview")
        col_prev_ctrl, col_prev_act = st.columns([2, 1])
        with col_prev_ctrl:
            if a_is_video and anim_duration > 0:
                preview_time = st.slider("Preview Timestamp (seconds)", 0.0, float(anim_duration), 0.0, 0.1, key="anim_prev_time")
            else:
                preview_time = 0.0
                st.caption("Single image selected — instant full preview available.")
        with col_prev_act:
            st.write("")
            if st.button("👁️ Render Instant Preview", key="anim_render_preview_btn", width="stretch"):
                with st.spinner("Generating live preview..."):
                    try:
                        p_config = AnimationConfig(
                            input_path=curr_anim_media,
                            direction=dir_choice,
                            style=style_choice,
                            strength=strength,
                            line_weight=line_weight,
                            color_saturation=color_saturation,
                            edge_soften=edge_soften,
                            film_grain=film_grain,
                            face_realism=face_realism,
                            face_fidelity=face_fidelity,
                        )
                        p_engine = AnimatorEngine(p_config)
                        orig_bgr, proc_bgr = p_engine.preview_frame(timestamp=preview_time)
                        # Convert BGR to RGB for Streamlit
                        st.session_state.anim_preview_orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                        st.session_state.anim_preview_proc = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        st.error(f"Preview failed: {e}")

        if st.session_state.anim_preview_orig is not None and st.session_state.anim_preview_proc is not None:
            st.markdown("##### Live Comparison")
            c_po, c_pp = st.columns(2)
            with c_po:
                st.caption("Original Source Frame")
                st.image(st.session_state.anim_preview_orig, width="stretch")
            with c_pp:
                st.caption(f"Transformed ({'Anime / Stylized' if dir_choice == 'real_to_anim' else 'Photorealistic'})")
                st.image(st.session_state.anim_preview_proc, width="stretch")

        st.divider()

        # Step 4: Output Destination & Encoding
        st.markdown("#### 4. Output & Execution")
        a_out_c1, a_out_cfmt, a_out_c2 = st.columns([2.6, 1.1, 1.3])
        with a_out_cfmt:
            if a_is_video:
                v_fmts = [".mp4", ".mkv", ".mov", ".avi"]
                c_v_fmt = Path(st.session_state.anim_output).suffix.lower() if st.session_state.anim_output else st.session_state.anim_video_format
                if c_v_fmt not in v_fmts:
                    c_v_fmt = ".mp4"
                v_f_idx = v_fmts.index(c_v_fmt)
                chosen_v_fmt = st.selectbox("Output Format", options=v_fmts, index=v_f_idx, format_func=lambda x: f"{x.upper()[1:]} ({x})", key="anim_v_fmt_drop")
                if chosen_v_fmt != c_v_fmt:
                    st.session_state.anim_video_format = chosen_v_fmt
                    if st.session_state.anim_output:
                        st.session_state.anim_output = str(Path(st.session_state.anim_output).with_suffix(chosen_v_fmt))
                    st.rerun()
            else:
                i_fmts = [".png", ".jpg", ".webp"]
                c_i_fmt = Path(st.session_state.anim_output).suffix.lower() if st.session_state.anim_output else st.session_state.anim_image_format
                if c_i_fmt not in i_fmts:
                    c_i_fmt = ".png"
                i_f_idx = i_fmts.index(c_i_fmt)
                chosen_i_fmt = st.selectbox("Output Format", options=i_fmts, index=i_f_idx, format_func=lambda x: f"{x.upper()[1:]} ({x})", key="anim_i_fmt_drop")
                if chosen_i_fmt != c_i_fmt:
                    st.session_state.anim_image_format = chosen_i_fmt
                    if st.session_state.anim_output:
                        st.session_state.anim_output = str(Path(st.session_state.anim_output).with_suffix(chosen_i_fmt))
                    st.rerun()

        with a_out_c1:
            anim_out_str = st.text_input("Output Destination:", value=st.session_state.anim_output, key="anim_out_text")
            st.session_state.anim_output = anim_out_str

        with a_out_c2:
            st.write("")
            st.write("")
            if st.button("💾 Save As...", key="anim_save_as_btn", width="stretch"):
                def_name = Path(anim_out_str).name if anim_out_str else f"{curr_anim_media.stem}_stylized{chosen_v_fmt if a_is_video else chosen_i_fmt}"
                save_c = select_save_file(
                    title="Choose Transformed File Destination",
                    default_name=def_name,
                    initial_dir=str(Path(anim_out_str).parent) if anim_out_str else None,
                )
                if save_c:
                    st.session_state.anim_output = save_c
                    st.rerun()

        # Codec for video
        if a_is_video:
            anim_codec = st.selectbox(
                "Video Encoder",
                options=["h264_nvenc", "hevc_nvenc", "libx264"],
                format_func=lambda x: {
                    "h264_nvenc": "H.264 (NVIDIA NVENC Fast)",
                    "hevc_nvenc": "HEVC/H.265 (NVIDIA NVENC)",
                    "libx264": "H.264 (CPU libx264)",
                }[x],
                key="anim_codec_select",
            )
        else:
            anim_codec = "h264_nvenc"

        btn_run_anim = st.button(
            "🚀 Transform Media (Real ⇄ Animation)",
            type="primary",
            key="run_anim_btn",
            width="stretch",
        )

        if btn_run_anim:
            if not anim_out_str:
                st.error("Please specify an output destination.")
            else:
                out_p = Path(anim_out_str).resolve()
                a_config = AnimationConfig(
                    input_path=curr_anim_media,
                    output_path=out_p,
                    direction=dir_choice,
                    style=style_choice,
                    strength=strength,
                    line_weight=line_weight,
                    color_saturation=color_saturation,
                    edge_soften=edge_soften,
                    film_grain=film_grain,
                    face_realism=face_realism,
                    face_fidelity=face_fidelity,
                    video_codec=anim_codec,
                )
                a_engine = AnimatorEngine(a_config)
                a_pbar = st.progress(0, text="Initializing Transformation...")
                a_status = st.empty()

                def anim_callback(stage_name: str, pct: float):
                    a_pbar.progress(int(pct), text=f"{stage_name} ({pct:.0f}%)")
                    a_status.info(f"🔄 **{stage_name}**")

                try:
                    start_t = time.time()
                    a_res = a_engine.execute(stage_callback=anim_callback)
                    elapsed_t = time.time() - start_t
                    a_pbar.progress(100, text="Transformation Complete!")

                    st.session_state.last_anim_result = {
                        "input": str(a_res.input_path),
                        "output": str(a_res.output_path),
                        "is_video": a_res.is_video,
                        "orig_res": f"{a_res.original_resolution[0]}x{a_res.original_resolution[1]}",
                        "proc_res": f"{a_res.processed_resolution[0]}x{a_res.processed_resolution[1]}",
                        "duration": a_res.duration,
                        "orig_fps": a_res.original_fps,
                        "direction": dir_choice,
                        "style": style_choice,
                        "elapsed": elapsed_t,
                    }
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Transformation failed: {e}")

    # Display Last Result
    if st.session_state.last_anim_result:
        a_last = st.session_state.last_anim_result
        st.markdown("---")
        st.markdown("### 🎉 Transformation Complete!")

        c_an_r1, c_an_r2, c_an_r3, c_an_r4 = st.columns(4)
        c_an_r1.metric("Resolution", a_last["proc_res"], f"from {a_last['orig_res']}")
        dir_display = "Real ➔ Animation" if a_last["direction"] == "real_to_anim" else "Animation ➔ Real"
        c_an_r2.metric("Direction", dir_display)
        c_an_r3.metric("Style", a_last["style"].title())
        c_an_r4.metric("Processing Time", f"{a_last['elapsed']:.1f} s")

        col_play_a, col_open_a, _ = st.columns([1, 1, 2])
        with col_play_a:
            if a_last["is_video"]:
                if st.button("▶️ Play Video", key="play_anim_btn", width="stretch"):
                    play_video(a_last["output"])
        with col_open_a:
            if st.button("📂 Show in Explorer", key="open_anim_exp", width="stretch"):
                open_in_file_explorer(a_last["output"])

        if not a_last["is_video"]:
            st.markdown("##### Before vs After")
            col_b_a, col_a_a = st.columns(2)
            with col_b_a:
                st.caption("Original Source")
                st.image(a_last["input"], width="stretch")
            with col_a_a:
                st.caption(f"Transformed Output ({dir_display})")
                st.image(a_last["output"], width="stretch")
        else:
            st.markdown("##### Output Video Player")
            a_out_f = Path(a_last["output"])
            if a_out_f.is_file():
                a_sz_mb = a_out_f.stat().st_size / (1024 * 1024)
                if a_sz_mb > 100:
                    st.info(f"💡 Large output video ({a_sz_mb:.1f} MB). Click **▶️ Play Video** to play instantly in your default media player.")
                    with st.expander("🌐 Preview in browser anyway"):
                        try:
                            st.video(a_last["output"])
                        except Exception as e:
                            st.warning(f"Browser player could not load video: {e}")
                else:
                    try:
                        st.video(a_last["output"])
                    except Exception as e:
                        st.warning(f"Browser player could not load video: {e}")


# ==============================================================================
# TAB 4: FACE STUDIO & REPLACEMENT
# ==============================================================================
with tab_face_studio:
    st.markdown("#### 👥 Video Face Analyzer & Replacement Studio")
    st.caption("Scan videos to detect and cluster all unique faces. Inspect thumbnails and appearances, then update or replace any detected face with custom pictures.")

    st.markdown("#### 1. Select Video or Photo")
    col_fs_pick, col_fs_clear = st.columns([1.5, 0.5])
    with col_fs_pick:
        if st.button("📁 Browse Video / Photo to Analyze...", key="fs_browse_btn", width="stretch"):
            chosen = select_media_file(title="Select Video or Photo for Face Analysis")
            if chosen:
                st.session_state.face_studio_input = chosen
                inp_p = Path(chosen)
                chosen_ext = inp_p.suffix.lower()
                target_ext = st.session_state.face_studio_video_format if chosen_ext in VIDEO_EXTENSIONS else st.session_state.face_studio_image_format
                st.session_state.face_studio_output = str(inp_p.parent / f"{inp_p.stem}_faceswapped{target_ext}")
                st.session_state.face_detected_clusters = []
                st.session_state.face_replacement_map = {}
                st.session_state.face_preview_orig = None
                st.session_state.face_preview_proc = None
                st.rerun()
    with col_fs_clear:
        if st.button("🧹 Clear", key="fs_clear_btn", width="stretch"):
            st.session_state.face_studio_input = ""
            st.session_state.face_studio_output = ""
            st.session_state.face_detected_clusters = []
            st.session_state.face_replacement_map = {}
            st.session_state.face_preview_orig = None
            st.session_state.face_preview_proc = None
            st.session_state.last_face_studio_result = None
            st.rerun()

    input_fs_str = st.text_input(
        "Or paste video path directly:",
        value=st.session_state.face_studio_input,
        placeholder="G:\\Videos\\interview.mp4",
        key="fs_input_text",
    )
    if input_fs_str and input_fs_str != st.session_state.face_studio_input:
        st.session_state.face_studio_input = input_fs_str
        inp_p = Path(input_fs_str)
        if inp_p.is_file():
            chosen_ext = inp_p.suffix.lower()
            target_ext = st.session_state.face_studio_video_format if chosen_ext in VIDEO_EXTENSIONS else st.session_state.face_studio_image_format
            st.session_state.face_studio_output = str(inp_p.parent / f"{inp_p.stem}_faceswapped{target_ext}")
            st.session_state.face_detected_clusters = []
            st.session_state.face_replacement_map = {}
            st.session_state.face_preview_orig = None
            st.session_state.face_preview_proc = None
            st.rerun()

    curr_fs_media = Path(st.session_state.face_studio_input)
    if not curr_fs_media.is_file():
        st.info("👋 Select a video or photo file above to scan and analyze all human faces.")
    else:
        fs_ext = curr_fs_media.suffix.lower()
        fs_is_image = fs_ext in IMAGE_EXTENSIONS
        fs_is_video = fs_ext in VIDEO_EXTENSIONS

        col_fs_prev, col_fs_meta = st.columns([1, 1])
        with col_fs_prev:
            st.markdown("##### Source Preview")
            if fs_is_image:
                st.image(str(curr_fs_media), width="stretch")
            elif fs_is_video:
                try:
                    if curr_fs_media.stat().st_size > 100 * 1024 * 1024:
                        st.caption(f"Source video ({curr_fs_media.stat().st_size / (1024*1024):.1f} MB)")
                    else:
                        st.video(str(curr_fs_media))
                except Exception as e:
                    st.caption(f"Video preview unavailable: {e}")
        with col_fs_meta:
            st.markdown("##### Source Properties")
            fs_duration = 0.0
            try:
                fs_meta = probe_video(curr_fs_media)
                fs_pv = fs_meta.primary_video
                st.text(f"File: {curr_fs_media.name}")
                st.text(f"Type: {'Image' if fs_is_image else 'Video'}")
                if fs_pv:
                    st.text(f"Resolution: {fs_pv.width} x {fs_pv.height}")
                    st.text(f"Framerate: {fs_pv.fps:.2f} fps")
                if fs_is_video:
                    fs_duration = fs_meta.duration or 0.0
                    total_f = int(fs_duration * (fs_pv.fps if fs_pv else 30.0))
                    st.text(f"Duration: {fs_duration:.1f} s ({fs_duration/60:.1f} min)")
                    st.text(f"Total Frames: ~{total_f:,}")
            except Exception as e:
                st.caption(f"Could not probe media: {e}")

        st.divider()

        # Step 2: Face Analysis
        st.markdown("#### 2. Analyze & Detect Faces in Video")
        col_scan_p1, col_scan_p2, col_scan_btn = st.columns([1.2, 1.2, 1.5])
        with col_scan_p1:
            scan_interval = st.selectbox(
                "Scanning Density",
                options=[0.25, 0.5, 1.0, 2.0],
                index=1,
                format_func=lambda x: f"Every {x} seconds" if x >= 1.0 else f"High Density (Every {x}s)",
                help="Shorter intervals catch brief face appearances; longer intervals scan faster.",
                key="fs_scan_interval_select",
            )
        with col_scan_p2:
            match_thresh = st.slider(
                "Face Clustering Sensitivity",
                min_value=0.30,
                max_value=0.60,
                value=0.40,
                step=0.05,
                help="Cosine similarity threshold for grouping faces belonging to the same person.",
                key="fs_match_thresh_slider",
            )
        with col_scan_btn:
            st.write("")
            btn_analyze_faces = st.button("🔍 Scan & Detect All Faces", type="primary", key="fs_run_scan_btn", width="stretch")

        if btn_analyze_faces:
            fs_cfg = FaceStudioConfig(
                input_path=curr_fs_media,
                similarity_threshold=match_thresh,
            )
            fs_eng = FaceStudioEngine(fs_cfg)
            fs_pbar = st.progress(0, text="Initializing Face Scanner...")
            fs_status = st.empty()

            def scan_callback(desc: str, pct: float):
                fs_pbar.progress(int(pct), text=f"{desc} ({pct:.0f}%)")
                fs_status.info(f"🔎 **{desc}**")

            try:
                found_clusters = fs_eng.analyze_faces(sample_interval=scan_interval, progress_cb=scan_callback)
                st.session_state.face_detected_clusters = found_clusters
                fs_pbar.progress(100, text=f"Scan complete: Found {len(found_clusters)} distinct individuals!")
                st.rerun()
            except Exception as e:
                st.error(f"Face analysis failed: {e}")

        # Display Detected Faces Gallery
        clusters = st.session_state.face_detected_clusters
        if clusters:
            st.markdown(f"### 🎯 Detected Faces ({len(clusters)} Distinct Individuals Found)")
            st.caption("Here are all the people detected across the video. Select any individual below to update their face with a new picture.")

            num_cols = min(4, max(1, len(clusters)))
            cols = st.columns(num_cols)
            for idx, c in enumerate(clusters):
                col = cols[idx % num_cols]
                with col:
                    thumb_rgb = cv2.cvtColor(c.thumbnail_bgr, cv2.COLOR_BGR2RGB)
                    st.image(thumb_rgb, width="stretch")
                    st.markdown(f"**{c.label}**")
                    st.caption(f"Appears in **{c.appearance_count}** sample frames\nFirst seen: `{c.first_seen_sec:.1f}s`")

            st.divider()

            # Step 3: Face Replacement Setup
            st.markdown("#### 3. Update / Replace Faces with New Pictures")
            st.markdown("Select a detected person, choose a replacement picture, and preview the new face.")

            col_tgt, col_pic = st.columns([1, 1.2])
            with col_tgt:
                person_options = [c.person_id for c in clusters]
                person_labels = {c.person_id: f"{c.label} (Seen in {c.appearance_count} frames, at {c.first_seen_sec:.1f}s)" for c in clusters}
                selected_person = st.selectbox(
                    "Select Person to Update:",
                    options=person_options,
                    format_func=lambda x: person_labels.get(x, x),
                    key="fs_selected_person",
                )

                target_cluster = next((c for c in clusters if c.person_id == selected_person), None)
                if target_cluster:
                    st.caption("Target face being replaced:")
                    st.image(cv2.cvtColor(target_cluster.thumbnail_bgr, cv2.COLOR_BGR2RGB), width=180)

            with col_pic:
                st.markdown("##### New Replacement Face Picture")
                c_pic_btn, c_pic_up = st.columns([1, 1])
                with c_pic_btn:
                    if st.button("🖼️ Browse Picture File...", key="fs_browse_pic_btn", width="stretch"):
                        pic_chosen = select_media_file(title="Select Replacement Face Picture")
                        if pic_chosen:
                            st.session_state.face_replacement_map[selected_person] = pic_chosen
                            st.session_state.face_preview_orig = None
                            st.session_state.face_preview_proc = None
                            st.rerun()

                uploaded_pic = st.file_uploader(
                    "Or upload image directly:",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"fs_file_upload_{selected_person}",
                )
                if uploaded_pic is not None:
                    temp_pic_dir = Path("bin/models/temp_faces")
                    temp_pic_dir.mkdir(parents=True, exist_ok=True)
                    temp_pic_path = temp_pic_dir / f"upload_{selected_person}_{uploaded_pic.name}"
                    with open(temp_pic_path, "wb") as f:
                        f.write(uploaded_pic.getbuffer())
                    st.session_state.face_replacement_map[selected_person] = str(temp_pic_path.resolve())

                current_rep_pic = st.session_state.face_replacement_map.get(selected_person, "")
                if current_rep_pic and Path(current_rep_pic).is_file():
                    st.caption("Active replacement picture:")
                    rep_img = cv2.imread(current_rep_pic)
                    if rep_img is not None:
                        st.image(cv2.cvtColor(rep_img, cv2.COLOR_BGR2RGB), width=180)
                else:
                    st.info("ℹ️ Select or upload a replacement photo showing the new person's face clearly.")

            c_tune1, c_tune2, c_tune3 = st.columns(3)
            with c_tune1:
                enable_gfp = st.checkbox("AI Facial Enhancement (GFPGAN)", value=True, help="Reconstructs photorealistic facial details and natural skin textures.")
            with c_tune2:
                gfp_fidelity = st.slider("Face Fidelity", 0.1, 1.0, 0.70, 0.05, help="Higher keeps more original person structure.")
            with c_tune3:
                blend_strength = st.slider("Replacement Strength", 0.1, 1.0, 1.0, 0.05, help="Opacity/blend factor of replaced face.")

            st.divider()

            # Step 4: Instant 1-Click Frame Preview
            st.markdown("#### 4. ⚡ Instant Frame Preview")
            col_fs_pv_ctrl, col_fs_pv_act = st.columns([2, 1])
            with col_fs_pv_ctrl:
                if fs_is_video and fs_duration > 0:
                    fs_prev_time = st.slider(
                        "Preview Timestamp (seconds)",
                        0.0,
                        float(fs_duration),
                        float(target_cluster.first_seen_sec if target_cluster else 0.0),
                        0.1,
                        key="fs_prev_time_slider",
                    )
                else:
                    fs_prev_time = 0.0
                    st.caption("Single image selected — instant full preview available.")
            with col_fs_pv_act:
                st.write("")
                if st.button("👁️ Preview Replaced Face on Frame", key="fs_render_prev_btn", width="stretch"):
                    active_pic = st.session_state.face_replacement_map.get(selected_person, "")
                    if not active_pic or not Path(active_pic).is_file():
                        st.warning("Please select or upload a replacement picture first.")
                    else:
                        with st.spinner("Generating live preview..."):
                            try:
                                pv_cfg = FaceStudioConfig(
                                    input_path=curr_fs_media,
                                    replacement_map={selected_person: Path(active_pic)},
                                    blend_strength=blend_strength,
                                    enable_gfpgan=enable_gfp,
                                    gfpgan_fidelity=gfp_fidelity,
                                    similarity_threshold=match_thresh,
                                )
                                pv_eng = FaceStudioEngine(pv_cfg)
                                orig_bgr, proc_bgr = pv_eng.preview_swap_frame(clusters, timestamp=fs_prev_time)
                                st.session_state.face_preview_orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                                st.session_state.face_preview_proc = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
                            except Exception as e:
                                st.error(f"Preview failed: {e}")

            if st.session_state.face_preview_orig is not None and st.session_state.face_preview_proc is not None:
                st.markdown("##### Live Comparison")
                c_fpo, c_fpp = st.columns(2)
                with c_fpo:
                    st.caption("Original Source Frame")
                    st.image(st.session_state.face_preview_orig, width="stretch")
                with c_fpp:
                    st.caption("Frame with Replaced Face (AI Enhanced)")
                    st.image(st.session_state.face_preview_proc, width="stretch")

            st.divider()

            # Step 5: Output Destination & Execute Full Video Face Swap
            st.markdown("#### 5. Output & Video Generation")
            fs_out_c1, fs_out_cfmt, fs_out_c2 = st.columns([2.6, 1.1, 1.3])
            with fs_out_cfmt:
                if fs_is_video:
                    v_fmts = [".mp4", ".mkv", ".mov", ".avi"]
                    c_v_fmt = Path(st.session_state.face_studio_output).suffix.lower() if st.session_state.face_studio_output else st.session_state.face_studio_video_format
                    if c_v_fmt not in v_fmts:
                        c_v_fmt = ".mp4"
                    v_f_idx = v_fmts.index(c_v_fmt)
                    chosen_v_fmt = st.selectbox("Output Format", options=v_fmts, index=v_f_idx, format_func=lambda x: f"{x.upper()[1:]} ({x})", key="fs_v_fmt_drop")
                    if chosen_v_fmt != c_v_fmt:
                        st.session_state.face_studio_video_format = chosen_v_fmt
                        if st.session_state.face_studio_output:
                            st.session_state.face_studio_output = str(Path(st.session_state.face_studio_output).with_suffix(chosen_v_fmt))
                        st.rerun()
                else:
                    i_fmts = [".png", ".jpg", ".webp"]
                    c_i_fmt = Path(st.session_state.face_studio_output).suffix.lower() if st.session_state.face_studio_output else st.session_state.face_studio_image_format
                    if c_i_fmt not in i_fmts:
                        c_i_fmt = ".png"
                    i_f_idx = i_fmts.index(c_i_fmt)
                    chosen_i_fmt = st.selectbox("Output Format", options=i_fmts, index=i_f_idx, format_func=lambda x: f"{x.upper()[1:]} ({x})", key="fs_i_fmt_drop")
                    if chosen_i_fmt != c_i_fmt:
                        st.session_state.face_studio_image_format = chosen_i_fmt
                        if st.session_state.face_studio_output:
                            st.session_state.face_studio_output = str(Path(st.session_state.face_studio_output).with_suffix(chosen_i_fmt))
                        st.rerun()

            with fs_out_c1:
                fs_out_str = st.text_input("Output Destination:", value=st.session_state.face_studio_output, key="fs_out_text")
                st.session_state.face_studio_output = fs_out_str

            with fs_out_c2:
                st.write("")
                st.write("")
                if st.button("💾 Save As...", key="fs_save_as_btn", width="stretch"):
                    def_name = Path(fs_out_str).name if fs_out_str else f"{curr_fs_media.stem}_faceswapped{chosen_v_fmt if fs_is_video else chosen_i_fmt}"
                    save_c = select_save_file(
                        title="Choose Face Swapped Destination",
                        default_name=def_name,
                        initial_dir=str(Path(fs_out_str).parent) if fs_out_str else None,
                    )
                    if save_c:
                        st.session_state.face_studio_output = save_c
                        st.rerun()

            if fs_is_video:
                fs_codec = st.selectbox(
                    "Video Encoder",
                    options=["h264_nvenc", "hevc_nvenc", "libx264"],
                    format_func=lambda x: {
                        "h264_nvenc": "H.264 (NVIDIA NVENC Fast)",
                        "hevc_nvenc": "HEVC/H.265 (NVIDIA NVENC)",
                        "libx264": "H.264 (CPU libx264)",
                    }[x],
                    key="fs_codec_select",
                )
            else:
                fs_codec = "h264_nvenc"

            btn_run_faceswap = st.button(
                "🚀 Update Faces Across Entire Video",
                type="primary",
                key="run_faceswap_btn",
                width="stretch",
            )

            if btn_run_faceswap:
                if not fs_out_str:
                    st.error("Please specify an output destination.")
                elif not any(Path(p).is_file() for p in st.session_state.face_replacement_map.values()):
                    st.error("Please select or upload at least one replacement picture for a detected person.")
                else:
                    out_p = Path(fs_out_str).resolve()
                    rep_map = {k: Path(v) for k, v in st.session_state.face_replacement_map.items() if Path(v).is_file()}

                    run_cfg = FaceStudioConfig(
                        input_path=curr_fs_media,
                        output_path=out_p,
                        replacement_map=rep_map,
                        blend_strength=blend_strength,
                        enable_gfpgan=enable_gfp,
                        gfpgan_fidelity=gfp_fidelity,
                        similarity_threshold=match_thresh,
                        video_codec=fs_codec,
                    )
                    run_eng = FaceStudioEngine(run_cfg)
                    run_pbar = st.progress(0, text="Initializing Face Replacement Pipeline...")
                    run_status = st.empty()

                    def run_callback(desc: str, pct: float):
                        run_pbar.progress(int(pct), text=f"{desc} ({pct:.0f}%)")
                        run_status.info(f"🔄 **{desc}**")

                    try:
                        start_t = time.time()
                        run_res = run_eng.execute_swap(clusters, stage_callback=run_callback)
                        elapsed_t = time.time() - start_t
                        run_pbar.progress(100, text="Face Replacement Complete!")

                        st.session_state.last_face_studio_result = {
                            "input": str(run_res.input_path),
                            "output": str(run_res.output_path),
                            "is_video": run_res.is_video,
                            "resolution": f"{run_res.resolution[0]}x{run_res.resolution[1]}",
                            "duration": run_res.duration,
                            "fps": run_res.fps,
                            "swapped_count": run_res.swapped_faces_count,
                            "elapsed": elapsed_t,
                        }
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Face replacement failed: {e}")

    # Display Last Result
    if st.session_state.last_face_studio_result:
        fs_last = st.session_state.last_face_studio_result
        st.markdown("---")
        st.markdown("### 🎉 Face Replacement Complete!")

        c_fs_r1, c_fs_r2, c_fs_r3, c_fs_r4 = st.columns(4)
        c_fs_r1.metric("Resolution", fs_last["resolution"])
        c_fs_r2.metric("Replaced Faces", f"{fs_last['swapped_count']} Individual(s)")
        if fs_last["is_video"] and fs_last["duration"]:
            c_fs_r3.metric("Video Duration", f"{fs_last['duration']:.1f} s")
        else:
            c_fs_r3.metric("Status", "Ready")
        c_fs_r4.metric("Processing Time", f"{fs_last['elapsed']:.1f} s")

        col_play_fs, col_open_fs, _ = st.columns([1, 1, 2])
        with col_play_fs:
            if fs_last["is_video"]:
                if st.button("▶️ Play Video", key="play_fs_btn", width="stretch"):
                    play_video(fs_last["output"])
        with col_open_fs:
            if st.button("📂 Show in Explorer", key="open_fs_exp", width="stretch"):
                open_in_file_explorer(fs_last["output"])

        if not fs_last["is_video"]:
            st.markdown("##### Before vs After")
            col_b_fs, col_a_fs = st.columns(2)
            with col_b_fs:
                st.caption("Original Source")
                st.image(fs_last["input"], width="stretch")
            with col_a_fs:
                st.caption("Faces Replaced (AI Restored)")
                st.image(fs_last["output"], width="stretch")
        else:
            st.markdown("##### Output Video Player")
            fs_out_f = Path(fs_last["output"])
            if fs_out_f.is_file():
                fs_sz_mb = fs_out_f.stat().st_size / (1024 * 1024)
                if fs_sz_mb > 100:
                    st.info(f"💡 Large output video ({fs_sz_mb:.1f} MB). Click **▶️ Play Video** to play instantly in your default media player.")
                    with st.expander("🌐 Preview in browser anyway"):
                        try:
                            st.video(fs_last["output"])
                        except Exception as e:
                            st.warning(f"Browser player could not load video: {e}")
                else:
                    try:
                        st.video(fs_last["output"])
                    except Exception as e:
                        st.warning(f"Browser player could not load video: {e}")


# ==============================================================================
# TAB 5: DIAGNOSTICS & SPECS
# ==============================================================================
with tab_diagnostics:


    st.markdown("#### 🔬 Detailed Media Stream Inspector")
    st.caption("Select any media file to inspect its full FFprobe stream details, codecs, bitrates, and packet formats.")

    if st.button("🔍 Select File to Inspect..."):
        chosen = select_media_file(title="Select File to Inspect")
        if chosen:
            st.session_state.inspect_file = chosen
            st.rerun()

    inspect_path = st.session_state.get("inspect_file", "")
    if inspect_path and Path(inspect_path).is_file():
        p = Path(inspect_path)
        try:
            meta = probe_video(p)
            st.success(f"Probed: **{p.name}**")
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Duration", f"{meta.duration:.2f} s")
            c_s2.metric("Size", f"{p.stat().st_size / (1024*1024):.2f} MB")
            c_s3.metric("Format", meta.format_name)

            st.markdown("##### Video Streams")
            for i, vs in enumerate(meta.video_streams):
                st.json({
                    "Codec": vs.codec_name,
                    "Resolution": f"{vs.width}x{vs.height}",
                    "FPS": vs.fps,
                    "Pixel Format": vs.pix_fmt,
                    "Bitrate": vs.bitrate,
                })

            st.markdown("##### Audio Streams")
            if not meta.audio_streams:
                st.info("No audio tracks detected in this file.")
            for i, as_ in enumerate(meta.audio_streams):
                st.json({
                    "Codec": as_.codec_name,
                    "Sample Rate": f"{as_.sample_rate} Hz",
                    "Channels": as_.channels,
                })
        except Exception as e:
            st.error(f"Error inspecting media: {e}")

