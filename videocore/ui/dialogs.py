"""Native Windows dialog helpers for VideoCore UI.

Provides seamless file and directory selection via native Windows File Explorer
without requiring users to upload multi-gigabyte files into web memory.
Runs file dialogs in an isolated main-thread GUI process (pythonw.exe) to prevent COM
apartment deadlocks and guarantee the dialog immediately pops up in the foreground.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional


def _get_gui_executable() -> str:
    """Returns pythonw.exe if available next to sys.executable, else sys.executable."""
    if sys.platform == "win32":
        pythonw = Path(sys.executable).parent / "pythonw.exe"
        if pythonw.is_file():
            return str(pythonw)
    return sys.executable


def _run_in_subprocess(action: str, **kwargs) -> Any:
    """Spawns an isolated Python GUI process to show the native dialog on the main thread."""
    script_path = Path(__file__).resolve()
    exe = _get_gui_executable()
    cmd = [exe, str(script_path), "--action", action]
    for k, v in kwargs.items():
        if v is not None:
            cmd.extend([f"--{k}", str(v)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except Exception as e:
        print(f"Dialog error: {e}", file=sys.stderr)
    return None


def select_folder(title: str = "Select Folder Containing Videos", initial_dir: Optional[str] = None) -> Optional[str]:
    """Opens a native directory selection dialog and returns the chosen folder path."""
    return _run_in_subprocess("folder", title=title, initial_dir=initial_dir)


def select_video_files(
    title: str = "Select Video Files to Join",
    initial_dir: Optional[str] = None,
) -> List[str]:
    """Opens a native multi-file selection dialog filtered for video formats."""
    res = _run_in_subprocess("video_files", title=title, initial_dir=initial_dir)
    return res if isinstance(res, list) else []


def select_media_file(
    title: str = "Select Media File to Enhance",
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    """Opens a native file selection dialog for images or videos."""
    return _run_in_subprocess("media_file", title=title, initial_dir=initial_dir)


def select_save_file(
    title: str = "Save Output Video As",
    default_name: str = "joined_output.mp4",
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    """Opens a native Save As dialog."""
    return _run_in_subprocess("save_file", title=title, default_name=default_name, initial_dir=initial_dir)


def open_in_file_explorer(target_path: str | Path) -> None:
    """Highlights the target file or directory in Windows File Explorer."""
    path = Path(target_path).resolve()
    if sys.platform == "win32":
        if path.is_file():
            subprocess.run(["explorer", f"/select,{str(path)}"], check=False)
        elif path.is_dir():
            os.startfile(str(path))
    else:
        parent = path.parent if path.is_file() else path
        subprocess.run(["xdg-open", str(parent)], check=False)


def play_video(target_path: str | Path) -> None:
    """Opens the video directly in the operating system's default media player."""
    path = Path(target_path).resolve()
    if not path.is_file():
        return
    if sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _show_dialog_main(action: str, title: str, initial_dir: Optional[str] = None, default_name: Optional[str] = None) -> Any:
    """Invoked in child GUI process where Tkinter runs safely on the main STA thread."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    # A tiny 1x1 transparent topmost window ensures proper focus and parenting on Windows
    root.geometry("1x1+0+0")
    root.attributes("-alpha", 0.01)
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.after(20, lambda: root.focus_force())

    res = None
    try:
        init_dir = initial_dir or str(Path.cwd())
        if action == "folder":
            folder = filedialog.askdirectory(
                parent=root,
                title=title,
                initialdir=init_dir,
                mustexist=True,
            )
            res = str(Path(folder).resolve()) if folder else None
        elif action == "video_files":
            file_types = [
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts *.mts *.m2ts *.3gp"),
                ("MP4 Videos", "*.mp4"),
                ("AVI Videos", "*.avi"),
                ("QuickTime MOV", "*.mov"),
                ("Matroska MKV", "*.mkv"),
                ("All Files", "*.*"),
            ]
            files = filedialog.askopenfilenames(
                parent=root,
                title=title,
                initialdir=init_dir,
                filetypes=file_types,
            )
            res = [str(Path(f).resolve()) for f in files] if files else []
        elif action == "media_file":
            file_types = [
                ("Supported Media", "*.jpg *.jpeg *.png *.webp *.bmp *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All Files", "*.*"),
            ]
            f = filedialog.askopenfilename(
                parent=root,
                title=title,
                initialdir=init_dir,
                filetypes=file_types,
            )
            res = str(Path(f).resolve()) if f else None
        elif action == "save_file":
            file_types = [
                ("MP4 Video", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("MKV Video", "*.mkv"),
                ("MOV Video", "*.mov"),
                ("All Files", "*.*"),
            ]
            f = filedialog.asksaveasfilename(
                parent=root,
                title=title,
                initialfile=default_name or "joined_output.mp4",
                initialdir=init_dir,
                filetypes=file_types,
            )
            res = str(Path(f).resolve()) if f else None
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native Dialog Child Process")
    parser.add_argument("--action", required=True, choices=["folder", "video_files", "media_file", "save_file"])
    parser.add_argument("--title", default="Select")
    parser.add_argument("--initial_dir", default=None)
    parser.add_argument("--default_name", default=None)
    args = parser.parse_args()

    result = _show_dialog_main(
        action=args.action,
        title=args.title,
        initial_dir=args.initial_dir,
        default_name=args.default_name,
    )
    print(json.dumps(result))
