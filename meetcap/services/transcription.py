"""speech-to-text transcription service using whisper"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class TranscriptSegment:
    """represents a single transcript segment with timing"""
    
    id: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    """complete transcription result"""
    
    audio_path: str
    sample_rate: int
    language: str
    segments: List[TranscriptSegment]
    duration: float
    stt: dict  # engine info


class TranscriptionService:
    """base class for transcription services"""
    
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """transcribe audio file to text."""
        raise NotImplementedError


class FasterWhisperService(TranscriptionService):
    """transcription using faster-whisper library"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None,
        auto_download: bool = True,
    ):
        """
        initialize faster-whisper service.
        
        args:
            model_path: local path to whisper model directory (optional if model_name provided)
            model_name: model name for auto-download (e.g., 'large-v3')
            device: device to use (auto, cpu, cuda, mps)
            compute_type: compute type (auto, int8, float16, float32)
            language: force language code (e.g., 'en') or none for auto-detect
            auto_download: whether to auto-download model if not found
        """
        self.model_name = model_name
        self.model_path = None
        self.auto_download = auto_download
        
        # if model_path provided, use it directly
        if model_path:
            self.model_path = Path(model_path).expanduser()
            # if path doesn't exist and we have a model name, we'll download
            if not self.model_path.exists() and model_name and auto_download:
                self.model_path = None  # will trigger download
        
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None
    
    def _load_model(self):
        """lazy load the model on first use."""
        if self.model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. "
                "install with: pip install faster-whisper"
            )
        
        # determine compute type
        compute_type = self.compute_type
        if compute_type == "auto":
            # use int8 for apple silicon (more compatible), float16 for others
            import platform
            if platform.processor() == "arm":
                compute_type = "int8"  # more compatible than int8_float16
            else:
                compute_type = "float16"
        
        # try loading model with fallback compute types
        model_source = None
        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
            local_only = True
            console.print(f"[cyan]loading whisper model from {self.model_path}...[/cyan]")
        elif self.model_name and self.auto_download:
            model_source = self.model_name
            local_only = False
            console.print(f"[cyan]loading whisper model '{self.model_name}'...[/cyan]")
        else:
            raise FileNotFoundError(
                f"model not found and auto-download disabled. "
                f"path: {self.model_path}, name: {self.model_name}"
            )
        
        # try compute types in order of preference
        compute_types_to_try = [compute_type] if compute_type != "auto" else ["int8", "float16", "float32"]
        last_error = None
        
        for ct in compute_types_to_try:
            try:
                if local_only:
                    self.model = WhisperModel(
                        model_source,
                        device=self.device,
                        compute_type=ct,
                        local_files_only=True,
                    )
                else:
                    self.model = WhisperModel(
                        model_source,
                        device=self.device,
                        compute_type=ct,
                        download_root=str(Path.home() / ".meetcap" / "models"),
                        local_files_only=False,
                    )
                console.print(f"[green]✓[/green] loaded with compute type: {ct}")
                break
            except Exception as e:
                last_error = e
                if "compute type" in str(e).lower():
                    console.print(f"[yellow]compute type {ct} not supported, trying next...[/yellow]")
                    continue
                else:
                    raise
        else:
            # all compute types failed
            raise RuntimeError(f"failed to load model with any compute type. last error: {last_error}")
    
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        transcribe audio file using faster-whisper.
        
        args:
            audio_path: path to audio file
            
        returns:
            transcription result with segments
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        
        # load model if needed
        self._load_model()
        
        console.print(f"[cyan]transcribing {audio_path.name}...[/cyan]")
        start_time = time.time()
        
        # run transcription
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("transcribing audio...", total=None)
            
            segments_list = []
            detected_language = self.language or "unknown"
            
            # transcribe with faster-whisper
            segments, info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                vad_filter=False,  # v1: no vad filtering
                word_timestamps=False,  # v1: segment-level only
                condition_on_previous_text=False,  # reduce hallucination
            )
            
            # collect segments
            for i, segment in enumerate(segments):
                segments_list.append(
                    TranscriptSegment(
                        id=i,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                    )
                )
                progress.update(
                    task,
                    description=f"transcribed {len(segments_list)} segments..."
                )
            
            detected_language = info.language
        
        # calculate duration
        duration = time.time() - start_time
        audio_duration = segments_list[-1].end if segments_list else 0.0
        
        console.print(
            f"[green]✓[/green] transcription complete: "
            f"{len(segments_list)} segments in {duration:.1f}s "
            f"(speed: {audio_duration/duration:.1f}x)"
        )
        
        return TranscriptResult(
            audio_path=str(audio_path),
            sample_rate=48000,  # we know our recording format
            language=detected_language,
            segments=segments_list,
            duration=audio_duration,
            stt={
                "engine": "faster-whisper",
                "model_path": str(self.model_path),
                "compute_type": self.compute_type,
            },
        )


class WhisperCppService(TranscriptionService):
    """transcription using whisper.cpp cli (alternative)"""
    
    def __init__(
        self,
        whisper_cpp_path: str,
        model_path: str,
        language: Optional[str] = None,
    ):
        """
        initialize whisper.cpp service.
        
        args:
            whisper_cpp_path: path to whisper.cpp main executable
            model_path: path to ggml model file
            language: force language code or none for auto-detect
        """
        self.whisper_cpp_path = Path(whisper_cpp_path)
        self.model_path = Path(model_path)
        
        if not self.whisper_cpp_path.exists():
            raise FileNotFoundError(f"whisper.cpp not found: {whisper_cpp_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        
        self.language = language
    
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        transcribe audio file using whisper.cpp.
        
        args:
            audio_path: path to audio file
            
        returns:
            transcription result with segments
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        
        console.print(f"[cyan]transcribing {audio_path.name} with whisper.cpp...[/cyan]")
        start_time = time.time()
        
        # build whisper.cpp command
        cmd = [
            str(self.whisper_cpp_path),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "--output-json",
            "--no-timestamps",  # we'll use srt for timing
            "--output-srt",
        ]
        
        if self.language:
            cmd.extend(["-l", self.language])
        
        # run whisper.cpp
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            
            # parse srt output for segments
            segments = self._parse_srt(result.stdout)
            
            duration = time.time() - start_time
            audio_duration = segments[-1].end if segments else 0.0
            
            console.print(
                f"[green]✓[/green] transcription complete: "
                f"{len(segments)} segments in {duration:.1f}s"
            )
            
            return TranscriptResult(
                audio_path=str(audio_path),
                sample_rate=48000,
                language=self.language or "auto",
                segments=segments,
                duration=audio_duration,
                stt={
                    "engine": "whisper.cpp",
                    "model_path": str(self.model_path),
                },
            )
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"whisper.cpp failed: {e.stderr}")
    
    def _parse_srt(self, srt_text: str) -> List[TranscriptSegment]:
        """parse srt format into segments."""
        segments = []
        lines = srt_text.strip().split("\n")
        i = 0
        segment_id = 0
        
        while i < len(lines):
            # skip segment number
            if lines[i].strip().isdigit():
                i += 1
                
                # parse timestamp line
                if i < len(lines) and " --> " in lines[i]:
                    times = lines[i].split(" --> ")
                    start = self._parse_srt_time(times[0])
                    end = self._parse_srt_time(times[1])
                    i += 1
                    
                    # collect text lines
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    segments.append(
                        TranscriptSegment(
                            id=segment_id,
                            start=start,
                            end=end,
                            text=" ".join(text_lines),
                        )
                    )
                    segment_id += 1
            i += 1
        
        return segments
    
    def _parse_srt_time(self, time_str: str) -> float:
        """convert srt timestamp to seconds."""
        # format: 00:00:00,000
        time_str = time_str.strip()
        parts = time_str.replace(",", ".").split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


def save_transcript(result: TranscriptResult, base_path: Path) -> tuple[Path, Path]:
    """
    save transcript to text and json files.
    
    args:
        result: transcription result
        base_path: base path without extension
        
    returns:
        tuple of (text_path, json_path)
    """
    # save plain text
    text_path = base_path.with_suffix(".transcript.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for segment in result.segments:
            f.write(f"{segment.text}\n")
    
    # save json with timestamps
    json_path = base_path.with_suffix(".transcript.json")
    json_data = {
        "audio_path": result.audio_path,
        "sample_rate": result.sample_rate,
        "language": result.language,
        "segments": [asdict(s) for s in result.segments],
        "duration": result.duration,
        "stt": result.stt,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓[/green] transcript saved:")
    console.print(f"  text: {text_path}")
    console.print(f"  json: {json_path}")
    
    return text_path, json_path