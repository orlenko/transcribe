"""FastAPI web application for speaker-aware transcription."""

import asyncio
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..database import Database
from ..env import load_transcribe_env
from ..models import DiarizedSegment
from ..timezone_utils import format_local_time, format_local_iso
from .config import Settings, get_settings
from .jobs import JobManager

# Initialize app
app = FastAPI(title="Transcribe Local", description="Speaker-aware local transcription")

# Setup paths
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Add custom Jinja filter for local time formatting
templates.env.filters["localtime"] = lambda dt, fmt="%Y-%m-%d %H:%M": format_local_time(dt, fmt)

# Global instances (initialized on startup)
db: Optional[Database] = None
job_manager: Optional[JobManager] = None
settings: Optional[Settings] = None
capture_sessions: dict[str, dict] = {}
capture_sessions_lock = asyncio.Lock()

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".mp4", ".opus"}
MIME_TYPE_TO_EXTENSION = {
    "audio/webm": ".webm",
    "audio/webm;codecs=opus": ".webm",
    "audio/ogg": ".ogg",
    "audio/ogg;codecs=opus": ".ogg",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/aac": ".aac",
}


def _sanitize_capture_label(label: Optional[str]) -> str:
    """Sanitize a user-provided capture label for safe filenames."""
    if not label:
        return ""

    cleaned = []
    for ch in label.lower().strip():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {"-", "_", " "}:
            cleaned.append("-")
    collapsed = "".join(cleaned).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed[:40]


def _extension_from_mime_type(mime_type: Optional[str]) -> str:
    """Infer file extension from a media MIME type."""
    if not mime_type:
        return ".webm"
    normalized = mime_type.lower().strip()
    return MIME_TYPE_TO_EXTENSION.get(normalized, ".webm")


def _safe_audio_name(filename: str, fallback_extension: str = ".mp3") -> str:
    """Return a filesystem-safe filename with a known audio extension."""
    safe_name = Path(filename).name or f"{uuid.uuid4()}{fallback_extension}"
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        safe_name = f"{Path(safe_name).stem or uuid.uuid4()}{fallback_extension}"
    return safe_name


def _unique_upload_path(upload_dir: Path, filename: str) -> Path:
    """Get a unique file path in upload_dir, preserving extension."""
    file_path = upload_dir / filename
    if not file_path.exists():
        return file_path

    stem = file_path.stem
    suffix = file_path.suffix
    counter = 1
    while file_path.exists():
        file_path = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return file_path


@app.on_event("startup")
async def startup():
    """Initialize database and job manager on startup."""
    global db, job_manager, settings
    load_transcribe_env()
    settings = get_settings()
    db = Database(settings.db_path)
    job_manager = JobManager(db, settings)


# ============== Pages ==============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, page: int = 1, per_page: int = 20):
    """Main dashboard page."""
    # Ensure valid pagination params
    page = max(1, page)
    per_page = max(1, min(100, per_page))  # Clamp between 1 and 100

    # Get paginated transcripts
    total = db.count_transcripts()
    offset = (page - 1) * per_page
    transcripts = db.list_transcripts(limit=per_page, offset=offset)
    speakers = db.list_speakers()
    jobs = job_manager.list_jobs()

    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "transcripts": transcripts,
        "speakers": speakers,
        "jobs": jobs,
        "settings": settings,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
        },
    })


@app.get("/transcript/{transcript_id}", response_class=HTMLResponse)
async def view_transcript(request: Request, transcript_id: int):
    """View a single transcript."""
    transcript = db.get_transcript(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    segments = db.get_segments(transcript_id)
    speakers = db.list_speakers()

    # Get audio file info
    transcripts_data = db.list_transcripts()
    audio_file = None
    for t, a in transcripts_data:
        if t.id == transcript_id:
            audio_file = a
            break

    # Get unique speaker labels in this transcript
    speaker_labels = list(set(seg.speaker_label for seg in segments if seg.speaker_label))

    return templates.TemplateResponse("transcript.html", {
        "request": request,
        "transcript": transcript,
        "segments": segments,
        "speakers": speakers,
        "speaker_labels": speaker_labels,
        "audio_file": audio_file,
    })


@app.get("/speakers", response_class=HTMLResponse)
async def speakers_page(request: Request):
    """Speaker management page."""
    speakers = db.list_speakers()
    return templates.TemplateResponse("speakers.html", {
        "request": request,
        "speakers": speakers,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": settings,
    })


@app.get("/capture", response_class=HTMLResponse)
async def capture_page(request: Request):
    """Browser recording page."""
    return templates.TemplateResponse("capture.html", {
        "request": request,
        "settings": settings,
    })


# ============== API Endpoints ==============

@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    mode: str = Form("hybrid"),
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    """Upload an audio file for transcription.

    Files are saved to the uploads folder and processed by the job runner daemon.
    """
    # Validate mode
    if mode not in ["hybrid", "openai", "local"]:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Save uploaded file - preserve original filename for timestamp extraction
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename to preserve timestamp info (e.g., recording-20240126-143045.mp3)
    fallback_ext = _extension_from_mime_type(file.content_type)
    original_name = _safe_audio_name(file.filename or f"{uuid.uuid4()}{fallback_ext}", fallback_extension=fallback_ext)
    file_path = _unique_upload_path(upload_dir, original_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # File is now in uploads folder - job runner will pick it up
    return {"status": "uploaded", "file_path": str(file_path), "message": "File uploaded. Job runner will process it."}


@app.post("/api/capture/start")
async def capture_start(
    mode: str = Form("hybrid"),
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    mime_type: Optional[str] = Form(None),
    label: Optional[str] = Form(None),
):
    """Start a browser recording session and return a chunk upload session id."""
    if mode not in {"hybrid", "openai", "local"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    session_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    capture_label = _sanitize_capture_label(label)
    ext = _extension_from_mime_type(mime_type)
    label_part = f"-{capture_label}" if capture_label else ""
    final_name = f"recording-{timestamp}{label_part}-browser-{session_id[:8]}{ext}"
    temp_path = upload_dir / f".capture-{session_id}.part"

    async with capture_sessions_lock:
        capture_sessions[session_id] = {
            "temp_path": str(temp_path),
            "final_name": final_name,
            "mode": mode,
            "language": language,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "mime_type": mime_type,
            "chunks": 0,
            "bytes": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

    return {
        "status": "started",
        "session_id": session_id,
        "file_name": final_name,
        "message": "Capture session started. Upload chunks to /api/capture/chunk.",
    }


@app.post("/api/capture/chunk")
async def capture_chunk(
    session_id: str = Form(...),
    chunk: UploadFile = File(...),
):
    """Append one MediaRecorder chunk to an active capture session."""
    chunk_data = await chunk.read()
    if not chunk_data:
        return {"status": "ignored", "reason": "empty chunk"}

    async with capture_sessions_lock:
        session = capture_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Capture session not found")

        temp_path = Path(session["temp_path"])
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "ab") as f:
            f.write(chunk_data)

        session["chunks"] += 1
        session["bytes"] += len(chunk_data)
        chunk_count = session["chunks"]
        byte_count = session["bytes"]

    return {
        "status": "ok",
        "session_id": session_id,
        "chunks": chunk_count,
        "bytes": byte_count,
    }


@app.post("/api/capture/finish")
async def capture_finish(session_id: str = Form(...)):
    """Finalize a browser recording session and move it into uploads."""
    async with capture_sessions_lock:
        session = capture_sessions.pop(session_id, None)

    if not session:
        raise HTTPException(status_code=404, detail="Capture session not found")

    temp_path = Path(session["temp_path"])
    if not temp_path.exists():
        raise HTTPException(status_code=400, detail="No data uploaded for this session")

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty capture file")

    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    final_name = _safe_audio_name(session["final_name"], fallback_extension=".webm")
    final_path = _unique_upload_path(upload_dir, final_name)
    temp_path.rename(final_path)

    return {
        "status": "uploaded",
        "file_path": str(final_path),
        "chunks": session["chunks"],
        "bytes": session["bytes"],
        "message": "Capture uploaded. Job runner will process it.",
    }


@app.post("/api/capture/cancel")
async def capture_cancel(session_id: str = Form(...)):
    """Cancel a browser recording session and remove temporary data."""
    async with capture_sessions_lock:
        session = capture_sessions.pop(session_id, None)

    if not session:
        return {"status": "not_found"}

    temp_path = Path(session["temp_path"])
    temp_path.unlink(missing_ok=True)
    return {"status": "cancelled"}


@app.get("/api/jobs")
async def list_jobs():
    """List all transcription jobs."""
    jobs = job_manager.list_jobs()
    return {"jobs": [j.to_dict() for j in jobs]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/transcripts")
async def list_transcripts_api(page: int = 1, per_page: int = 20):
    """List transcripts with pagination."""
    page = max(1, page)
    per_page = max(1, min(100, per_page))

    total = db.count_transcripts()
    offset = (page - 1) * per_page
    transcripts = db.list_transcripts(limit=per_page, offset=offset)

    return {
        "transcripts": [
            {
                "id": t.id,
                "audio_id": t.audio_id,
                "file_path": a.file_path,
                "file_name": Path(a.file_path).name,
                "duration": a.duration_seconds,
                "model": t.model_name,
                "language": t.language,
                "title": t.title,
                "created_at": format_local_iso(t.created_at),
            }
            for t, a in transcripts
        ],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page if total > 0 else 1,
        },
    }


@app.get("/api/transcripts/{transcript_id}")
async def get_transcript(transcript_id: int):
    """Get transcript with segments."""
    transcript = db.get_transcript(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    segments = db.get_segments(transcript_id)

    return {
        "transcript": {
            "id": transcript.id,
            "audio_id": transcript.audio_id,
            "model": transcript.model_name,
            "language": transcript.language,
            "created_at": format_local_iso(transcript.created_at),
        },
        "segments": [
            {
                "id": seg.id,
                "start": seg.start_time,
                "end": seg.end_time,
                "text": seg.text,
                "speaker_label": seg.speaker_label,
                "speaker_id": seg.speaker_id,
            }
            for seg in segments
        ],
    }


@app.delete("/api/transcripts/{transcript_id}")
async def delete_transcript(transcript_id: int):
    """Delete a transcript."""
    if db.delete_transcript(transcript_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Transcript not found")


@app.post("/api/transcripts/{transcript_id}/assign")
async def assign_speaker(
    transcript_id: int,
    speaker_label: str = Form(...),
    speaker_name: str = Form(...),
    learn_voice: bool = Form(True),
):
    """Assign a speaker name to a speaker label and optionally learn their voice."""
    transcript = db.get_transcript(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    # Get or create speaker
    speaker = db.get_speaker(speaker_name)
    if not speaker:
        speaker = db.add_speaker(speaker_name)

    # Assign speaker to segments
    updated = db.assign_speaker_to_label(transcript_id, speaker_label, speaker.id)

    result = {
        "status": "assigned",
        "speaker_id": speaker.id,
        "segments_updated": updated,
        "embeddings_learned": 0,
    }

    # Learn voice embeddings if requested
    if learn_voice and updated > 0:
        try:
            # Get audio file path
            transcripts_data = db.list_transcripts()
            audio_path = None
            audio_id = None
            for t, a in transcripts_data:
                if t.id == transcript_id:
                    audio_path = a.file_path
                    audio_id = a.id
                    break

            if audio_path and Path(audio_path).exists():
                from ..diarizer import Diarizer
                from ..speaker_matcher import SpeakerMatcher

                diarizer = Diarizer()
                matcher = SpeakerMatcher(db)

                segments = db.get_segments_by_speaker_label(transcript_id, speaker_label)
                if segments:
                    diarized = [
                        DiarizedSegment(
                            start=seg.start_time,
                            end=seg.end_time,
                            text=seg.text,
                            speaker_label=seg.speaker_label or speaker_label,
                        )
                        for seg in segments
                    ]

                    embeddings = diarizer.extract_embeddings_for_speaker(
                        audio_path, diarized, speaker_label, max_embeddings=5
                    )

                    if embeddings:
                        count = matcher.learn_speaker(speaker.id, embeddings, audio_id)
                        result["embeddings_learned"] = count
        except Exception as e:
            result["embedding_error"] = str(e)

    return result


@app.post("/api/segments/{segment_id}/assign")
async def assign_segment_speaker(
    segment_id: int,
    speaker_name: str = Form(...),
):
    """Assign a speaker to a single segment (not all with same label)."""
    # Get or create speaker
    speaker = db.get_speaker(speaker_name)
    if not speaker:
        speaker = db.add_speaker(speaker_name)

    # Update just this segment
    if db.update_segment_speaker(segment_id, speaker.id):
        return {
            "status": "assigned",
            "speaker_id": speaker.id,
            "speaker_name": speaker.name,
        }
    raise HTTPException(status_code=404, detail="Segment not found")


@app.get("/api/speakers")
async def list_speakers_api():
    """List all speakers."""
    speakers = db.list_speakers()
    return {
        "speakers": [
            {
                "id": s.id,
                "name": s.name,
                "sample_count": s.sample_count,
                "notes": s.notes,
                "created_at": format_local_iso(s.created_at),
            }
            for s in speakers
        ]
    }


@app.post("/api/speakers")
async def create_speaker(
    name: str = Form(...),
    notes: Optional[str] = Form(None),
):
    """Create a new speaker."""
    if db.get_speaker(name):
        raise HTTPException(status_code=400, detail="Speaker already exists")

    speaker = db.add_speaker(name, notes)
    return {
        "id": speaker.id,
        "name": speaker.name,
        "sample_count": speaker.sample_count,
    }


@app.delete("/api/speakers/{speaker_name}")
async def delete_speaker(speaker_name: str):
    """Delete a speaker."""
    if db.delete_speaker(speaker_name):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Speaker not found")


@app.post("/api/speakers/{old_name}/rename")
async def rename_speaker(old_name: str, new_name: str = Form(...)):
    """Rename a speaker."""
    if not db.get_speaker(old_name):
        raise HTTPException(status_code=404, detail="Speaker not found")
    if db.get_speaker(new_name):
        raise HTTPException(status_code=400, detail="New name already exists")

    db.rename_speaker(old_name, new_name)
    return {"status": "renamed", "old_name": old_name, "new_name": new_name}


@app.get("/api/export/{transcript_id}")
async def export_transcript(
    transcript_id: int,
    format: str = "txt",
):
    """Export transcript in various formats."""
    transcript = db.get_transcript(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    segments = db.get_segments(transcript_id)
    if not segments:
        raise HTTPException(status_code=404, detail="No segments found")

    # Convert to DiarizedSegment for formatting
    diarized = [
        DiarizedSegment(
            start=seg.start_time,
            end=seg.end_time,
            text=seg.text,
            speaker_label=seg.speaker_label or "UNKNOWN",
        )
        for seg in segments
    ]

    from ..transcriber import segments_to_srt, segments_to_text, segments_to_vtt
    import json

    if format == "json":
        content = json.dumps([
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker_label, "text": seg.text}
            for seg in diarized
        ], indent=2)
        media_type = "application/json"
        ext = "json"
    elif format == "srt":
        content = segments_to_srt(diarized)
        media_type = "text/plain"
        ext = "srt"
    elif format == "vtt":
        content = segments_to_vtt(diarized)
        media_type = "text/vtt"
        ext = "vtt"
    else:
        content = segments_to_text(diarized)
        media_type = "text/plain"
        ext = "txt"

    # Create temp file for download
    with tempfile.NamedTemporaryFile(mode="w", suffix=f".{ext}", delete=False) as f:
        f.write(content)
        temp_path = f.name

    return FileResponse(
        temp_path,
        media_type=media_type,
        filename=f"transcript_{transcript_id}.{ext}",
    )


@app.get("/api/audio/{transcript_id}")
async def get_audio(transcript_id: int):
    """Serve the audio file for a transcript."""
    transcript = db.get_transcript(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    # Get audio file path
    transcripts_data = db.list_transcripts()
    audio_path = None
    for t, a in transcripts_data:
        if t.id == transcript_id:
            audio_path = a.file_path
            break

    if not audio_path:
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Check if file exists at original path
    audio_file = Path(audio_path)
    if not audio_file.exists():
        # Check in processed folder (file may have been moved after transcription)
        processed_dir = settings.data_dir / "processed"
        processed_file = processed_dir / audio_file.name
        if processed_file.exists():
            audio_file = processed_file
        else:
            # Also check with potential timestamp suffix (e.g., filename_1234567890.mp3)
            stem = audio_file.stem
            suffix = audio_file.suffix
            for f in processed_dir.glob(f"{stem}*{suffix}"):
                audio_file = f
                break
            else:
                raise HTTPException(status_code=404, detail="Audio file not found")

    # Determine media type based on extension
    ext = audio_file.suffix.lower()
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
    }
    media_type = media_types.get(ext, "audio/mpeg")

    return FileResponse(str(audio_file), media_type=media_type)


@app.post("/api/settings")
async def update_settings(
    transcription_mode: str = Form(...),
    openai_model: str = Form("whisper-1"),
    local_model: str = Form("large-v3"),
    default_language: Optional[str] = Form(None),
    server_url: Optional[str] = Form(None),
):
    """Update application settings."""
    global settings

    if transcription_mode not in {"hybrid", "openai", "local"}:
        raise HTTPException(status_code=400, detail="Invalid transcription mode")

    if server_url is not None:
        server_url = server_url.strip()
        if not server_url:
            raise HTTPException(status_code=400, detail="Server URL cannot be empty")
        if not (server_url.startswith("http://") or server_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="Server URL must start with http:// or https://")
        settings.server_url = server_url

    settings.transcription_mode = transcription_mode
    settings.openai_model = openai_model
    settings.local_model = local_model
    settings.default_language = default_language or None

    settings.save()

    return {"status": "saved"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
