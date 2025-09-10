"""speech-to-text transcription service using whisper"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
    speaker_id: int | None = None  # speaker identifier for diarization
    confidence: float | None = None  # recognition confidence score


@dataclass
class TranscriptResult:
    """complete transcription result"""

    audio_path: str
    sample_rate: int
    language: str
    segments: list[TranscriptSegment]
    duration: float
    stt: dict  # engine info
    speakers: list[dict] | None = None  # speaker metadata for diarization
    diarization_enabled: bool = False  # flag indicating if diarization was used


class TranscriptionService:
    """base class for transcription services"""

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """transcribe audio file to text."""
        raise NotImplementedError

    def load_model(self) -> None:
        """explicitly load model into memory."""
        # default implementation calls _load_model if it exists
        if hasattr(self, "_load_model"):
            self._load_model()

    def unload_model(self) -> None:
        """explicitly unload model from memory with cleanup."""
        raise NotImplementedError

    def is_loaded(self) -> bool:
        """check if model is currently loaded in memory."""
        return getattr(self, "model", None) is not None

    def get_memory_usage(self) -> dict:
        """return current memory usage statistics."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Physical memory
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


class FasterWhisperService(TranscriptionService):
    """transcription using faster-whisper library"""

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
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
        except ImportError as e:
            raise ImportError(
                "faster-whisper not installed. install with: pip install faster-whisper"
            ) from e

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
        compute_types_to_try = (
            [compute_type] if compute_type != "auto" else ["int8", "float16", "float32"]
        )
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
                    console.print(
                        f"[yellow]compute type {ct} not supported, trying next...[/yellow]"
                    )
                    continue
                else:
                    raise
        else:
            # all compute types failed
            raise RuntimeError(
                f"failed to load model with any compute type. last error: {last_error}"
            )

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
                progress.update(task, description=f"transcribed {len(segments_list)} segments...")

            detected_language = info.language

        # calculate duration
        duration = time.time() - start_time
        audio_duration = segments_list[-1].end if segments_list else 0.0

        console.print(
            f"[green]✓[/green] transcription complete: "
            f"{len(segments_list)} segments in {duration:.1f}s "
            f"(speed: {audio_duration / duration:.1f}x)"
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

    def unload_model(self) -> None:
        """unload faster-whisper model and cleanup GPU/CPU resources."""
        if self.model is not None:
            # Release model reference
            del self.model
            self.model = None

            # Force garbage collection
            import gc

            gc.collect()

            # Clear GPU cache if using CUDA
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            console.print("[dim]faster-whisper model unloaded[/dim]")


class WhisperCppService(TranscriptionService):
    """transcription using whisper.cpp cli (alternative)"""

    def __init__(
        self,
        whisper_cpp_path: str,
        model_path: str,
        language: str | None = None,
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
            "-m",
            str(self.model_path),
            "-f",
            str(audio_path),
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
            raise RuntimeError(f"whisper.cpp failed: {e.stderr}") from e

    def unload_model(self) -> None:
        """unload WhisperCpp service resources."""
        # WhisperCpp is CLI-based, no models to unload
        # Just ensure paths are cleared
        self.whisper_cpp_path = None
        self.model_path = None

        # Force garbage collection for any cached data
        import gc

        gc.collect()

        console.print("[dim]whisper.cpp resources cleared[/dim]")

    def _parse_srt(self, srt_text: str) -> list[TranscriptSegment]:
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


class MlxWhisperService(TranscriptionService):
    """transcription using mlx-whisper library (Apple Silicon optimized)"""

    def __init__(
        self,
        model_name: str = "mlx-community/whisper-large-v3-turbo",
        model_path: str | None = None,
        language: str | None = None,
        auto_download: bool = True,
    ):
        """
        initialize mlx-whisper service.

        args:
            model_name: hugging face model name (e.g., 'mlx-community/whisper-large-v3-turbo')
            model_path: local path to model directory (optional)
            language: force language code (e.g., 'en') or none for auto-detect
            auto_download: whether to auto-download model if not found
        """
        self.model_name = model_name
        self.model_path = None
        self.auto_download = auto_download
        self.language = language
        self.model = None
        self.model_source = None

        # if model_path provided, use it directly
        if model_path:
            self.model_path = Path(model_path).expanduser()

    def _load_model(self):
        """lazy load the model on first use."""
        if self.model is not None:
            return

        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mlx-whisper not installed. install with: pip install mlx-whisper"
            ) from e

        # determine model source
        if self.model_path and self.model_path.exists():
            self.model_source = str(self.model_path)
            console.print(f"[cyan]using mlx-whisper model from {self.model_path}...[/cyan]")
        elif self.auto_download:
            self.model_source = self.model_name
            console.print(f"[cyan]using mlx-whisper model '{self.model_name}'...[/cyan]")
        else:
            raise FileNotFoundError(
                f"mlx-whisper model not found and auto-download disabled. "
                f"path: {self.model_path}, name: {self.model_name}"
            )

        # mlx-whisper doesn't have a separate model loading step
        # model loading happens automatically during transcription
        self.model = "loaded"  # just mark as ready
        console.print("[green]✓[/green] mlx-whisper model ready")

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        transcribe audio file using mlx-whisper.

        args:
            audio_path: path to audio file

        returns:
            transcription result with segments
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        # load model if needed
        self._load_model()

        console.print(f"[cyan]transcribing {audio_path.name} with mlx-whisper...[/cyan]")
        start_time = time.time()

        # run transcription
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("transcribing audio...", total=None)

            try:
                import mlx_whisper

                # transcribe with mlx-whisper
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=self.model_source,
                    language=self.language,
                    word_timestamps=True,  # enable word-level timestamps
                )

                progress.update(task, description="processing segments...")

                # convert mlx-whisper output to our format
                segments_list = []
                if "segments" in result:
                    for i, segment in enumerate(result["segments"]):
                        segments_list.append(
                            TranscriptSegment(
                                id=i,
                                start=segment.get("start", 0.0),
                                end=segment.get("end", 0.0),
                                text=segment.get("text", "").strip(),
                            )
                        )
                else:
                    # fallback: create single segment from full text
                    segments_list.append(
                        TranscriptSegment(
                            id=0,
                            start=0.0,
                            end=0.0,  # we don't know the duration
                            text=result.get("text", "").strip(),
                        )
                    )

                detected_language = result.get("language", "unknown")

            except Exception as e:
                console.print(f"[red]mlx-whisper transcription failed: {e}[/red]")
                # fallback to faster-whisper if available
                console.print("[yellow]falling back to faster-whisper...[/yellow]")
                try:
                    fallback_service = FasterWhisperService(
                        model_name="large-v3",
                        language=self.language,
                        auto_download=True,
                    )
                    return fallback_service.transcribe(audio_path)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"both mlx-whisper and faster-whisper failed. "
                        f"mlx error: {e}, fallback error: {fallback_error}"
                    ) from e

        # calculate duration
        duration = time.time() - start_time
        audio_duration = segments_list[-1].end if segments_list else 0.0

        console.print(
            f"[green]✓[/green] mlx-whisper transcription complete: "
            f"{len(segments_list)} segments in {duration:.1f}s "
            f"(speed: {audio_duration / duration:.1f}x)"
            if audio_duration > 0
            else ""
        )

        return TranscriptResult(
            audio_path=str(audio_path),
            sample_rate=48000,  # we know our recording format
            language=detected_language,
            segments=segments_list,
            duration=audio_duration,
            stt={
                "engine": "mlx-whisper",
                "model_name": self.model_name,
                "model_path": str(self.model_path) if self.model_path else None,
            },
        )

    def unload_model(self) -> None:
        """unload MLX-whisper model and cleanup Metal resources."""
        if self.model is not None:
            # MLX models don't have explicit cleanup, but we can clear references
            del self.model
            self.model = None
            self.model_source = None

            # Clear MLX memory if available
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except (ImportError, AttributeError):
                pass

            # Force garbage collection
            import gc

            gc.collect()

            console.print("[dim]mlx-whisper model unloaded[/dim]")


class VoskTranscriptionService(TranscriptionService):
    """transcription using vosk with speaker diarization support"""

    def __init__(
        self,
        model_path: str,
        spk_model_path: str | None = None,
        sample_rate: int = 48000,
        enable_diarization: bool = False,
    ):
        """
        initialize vosk service.

        args:
            model_path: path to vosk model directory
            spk_model_path: path to speaker model directory (optional)
            sample_rate: expected audio sample rate
            enable_diarization: enable speaker identification
        """
        self.model_path = Path(model_path).expanduser()
        self.spk_model_path = Path(spk_model_path).expanduser() if spk_model_path else None
        self.sample_rate = sample_rate
        self.enable_diarization = enable_diarization and spk_model_path is not None
        self.model = None
        self.spk_model = None

    def _load_model(self):
        """lazy load the vosk model on first use."""
        if self.model is not None:
            return

        try:
            import vosk
        except ImportError as e:
            raise ImportError("vosk not installed. install with: pip install vosk") from e

        if not self.model_path.exists():
            raise FileNotFoundError(f"vosk model not found: {self.model_path}")

        console.print(f"[cyan]loading vosk model from {self.model_path.name}...[/cyan]")

        try:
            # set log level to warnings only
            vosk.SetLogLevel(-1)

            # load main recognition model
            self.model = vosk.Model(str(self.model_path))
            console.print("[green]✓[/green] vosk model loaded")

            # load speaker model if diarization enabled
            if self.enable_diarization and self.spk_model_path:
                if self.spk_model_path.exists():
                    console.print(
                        f"[cyan]loading speaker model from {self.spk_model_path.name}...[/cyan]"
                    )
                    self.spk_model = vosk.SpkModel(str(self.spk_model_path))
                    console.print("[green]✓[/green] speaker model loaded")
                else:
                    console.print(
                        "[yellow]warning: speaker model not found, diarization disabled[/yellow]"
                    )
                    self.enable_diarization = False

        except Exception as e:
            raise RuntimeError(f"failed to load vosk model: {e}") from e

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """
        transcribe audio file using vosk.

        args:
            audio_path: path to audio file

        returns:
            transcription result with segments and optional speaker info
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        # load model if needed
        self._load_model()

        console.print(f"[cyan]transcribing {audio_path.name} with vosk...[/cyan]")
        start_time = time.time()

        try:
            import tempfile

            import soundfile as sf
            import vosk

            # check if we need to convert the audio format
            # vosk (via soundfile) only supports WAV, FLAC, OGG, etc. - not M4A/MP4
            supported_extensions = {".wav", ".flac", ".ogg", ".opus", ".raw"}
            needs_conversion = audio_path.suffix.lower() not in supported_extensions

            audio_to_process = audio_path
            temp_wav_file = None

            if needs_conversion:
                console.print(f"[yellow]converting {audio_path.suffix} to WAV for vosk...[/yellow]")

                # create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_wav_file = Path(temp_file.name)

                # convert using ffmpeg
                try:
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(audio_path),
                        "-ac",
                        "1",  # mono
                        "-ar",
                        str(self.sample_rate),  # resample to expected rate
                        "-f",
                        "wav",
                        "-y",  # overwrite
                        str(temp_wav_file),
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                    audio_to_process = temp_wav_file
                    console.print("[green]✓[/green] audio converted to WAV")

                except subprocess.CalledProcessError as e:
                    if temp_wav_file and temp_wav_file.exists():
                        temp_wav_file.unlink()
                    raise RuntimeError(f"ffmpeg conversion failed: {e.stderr}") from e

            # read audio file
            audio_data, file_sample_rate = sf.read(str(audio_to_process))

            # handle stereo to mono conversion if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # convert to int16 for vosk
            import numpy as np

            audio_int16 = (audio_data * 32767).astype(np.int16)

            # create recognizer
            rec = vosk.KaldiRecognizer(self.model, file_sample_rate)
            rec.SetWords(True)  # enable word-level timestamps

            # enable speaker diarization if available
            if self.enable_diarization and self.spk_model:
                rec.SetSpkModel(self.spk_model)

            # process audio in chunks
            chunk_size = 4000  # samples per chunk
            segments_list = []
            speaker_embeddings = []
            chunk_results = []  # store intermediate results with speaker vectors

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("processing audio...", total=len(audio_int16))

                for i in range(0, len(audio_int16), chunk_size):
                    chunk = audio_int16[i : i + chunk_size]
                    if rec.AcceptWaveform(chunk.tobytes()):
                        # got a complete result for this chunk
                        result = json.loads(rec.Result())
                        if result:
                            chunk_results.append(result)
                            # collect speaker embedding if present
                            if self.enable_diarization and "spk" in result:
                                speaker_embeddings.append(result["spk"])
                    progress.update(task, completed=min(i + chunk_size, len(audio_int16)))

                # get final result
                final_result = json.loads(rec.FinalResult())
                if final_result:
                    chunk_results.append(final_result)
                    if self.enable_diarization and "spk" in final_result:
                        speaker_embeddings.append(final_result["spk"])

            # process all results into segments with speaker assignment
            all_words = []
            chunk_to_spk_id = {}  # map chunk index to speaker ID

            # extract all words from all chunks
            for chunk_idx, result in enumerate(chunk_results):
                if "result" in result:
                    for word_info in result["result"]:
                        word_info["chunk_idx"] = chunk_idx
                        all_words.append(word_info)

            # cluster speaker embeddings if we have them
            if self.enable_diarization and speaker_embeddings:
                try:
                    # perform speaker clustering
                    from sklearn.cluster import AgglomerativeClustering
                    from sklearn.preprocessing import normalize

                    # convert to numpy array and normalize
                    X = np.array(speaker_embeddings, dtype=np.float32)
                    Xn = normalize(X, norm="l2")

                    # estimate number of speakers (2-4 typical for meetings)
                    # could be made configurable
                    n_speakers = min(4, max(2, len(speaker_embeddings) // 10))

                    # cluster using cosine distance
                    clustering = AgglomerativeClustering(
                        n_clusters=n_speakers, metric="cosine", linkage="average"
                    )
                    labels = clustering.fit_predict(Xn)

                    # map chunk indices to speaker IDs
                    for chunk_idx, spk_id in enumerate(labels):
                        chunk_to_spk_id[chunk_idx] = int(spk_id)

                    console.print(f"[dim]identified {n_speakers} speakers[/dim]")

                except ImportError:
                    console.print(
                        "[yellow]scikit-learn not installed, speaker clustering disabled[/yellow]"
                    )
                    self.enable_diarization = False
                except Exception as e:
                    console.print(f"[yellow]speaker clustering failed: {e}[/yellow]")
                    self.enable_diarization = False

            # group words into segments with speaker IDs
            for word_info in all_words:
                # determine speaker ID for this word
                speaker_id = None
                if self.enable_diarization and word_info["chunk_idx"] in chunk_to_spk_id:
                    speaker_id = chunk_to_spk_id[word_info["chunk_idx"]]

                # group words into segments by pauses
                if not segments_list or (word_info["start"] - segments_list[-1].end > 0.5):
                    # new segment
                    segment = TranscriptSegment(
                        id=len(segments_list),
                        start=word_info["start"],
                        end=word_info["end"],
                        text=word_info["word"],
                        confidence=word_info.get("conf", 1.0),
                        speaker_id=speaker_id,
                    )
                    segments_list.append(segment)
                else:
                    # append to current segment (keep same speaker)
                    segments_list[-1].text += " " + word_info["word"]
                    segments_list[-1].end = word_info["end"]
                    # update speaker if different (shouldn't happen within same segment)
                    if segments_list[-1].speaker_id != speaker_id:
                        # actually start a new segment for speaker change
                        segment = TranscriptSegment(
                            id=len(segments_list),
                            start=word_info["start"],
                            end=word_info["end"],
                            text=word_info["word"],
                            confidence=word_info.get("conf", 1.0),
                            speaker_id=speaker_id,
                        )
                        segments_list.append(segment)

            # extract speaker info if available
            speakers = None
            if self.enable_diarization and chunk_to_spk_id:
                # create speaker metadata
                unique_speakers = set(chunk_to_spk_id.values())
                speakers = [
                    {"id": spk_id, "label": f"Speaker {spk_id + 1}"} for spk_id in unique_speakers
                ]
                console.print(
                    f"[dim]speaker diarization complete: {len(unique_speakers)} speakers[/dim]"
                )

        except ImportError as e:
            # clean up temp file if created
            if temp_wav_file and temp_wav_file.exists():
                temp_wav_file.unlink()

            console.print(f"[red]vosk dependencies missing: {e}[/red]")
            console.print("[yellow]falling back to faster-whisper...[/yellow]")
            # fallback to whisper
            fallback_service = FasterWhisperService(
                model_name="large-v3",
                auto_download=True,
            )
            return fallback_service.transcribe(audio_path)

        except Exception as e:
            # clean up temp file if created
            if temp_wav_file and temp_wav_file.exists():
                temp_wav_file.unlink()

            console.print(f"[red]vosk transcription failed: {e}[/red]")
            console.print("[yellow]falling back to faster-whisper...[/yellow]")
            # fallback to whisper
            fallback_service = FasterWhisperService(
                model_name="large-v3",
                auto_download=True,
            )
            return fallback_service.transcribe(audio_path)

        finally:
            # always clean up temp file if created
            if temp_wav_file and temp_wav_file.exists():
                try:
                    temp_wav_file.unlink()
                except OSError:
                    pass  # ignore cleanup errors

        # calculate duration
        duration = time.time() - start_time
        audio_duration = segments_list[-1].end if segments_list else 0.0

        console.print(
            f"[green]✓[/green] vosk transcription complete: "
            f"{len(segments_list)} segments in {duration:.1f}s "
            f"(speed: {audio_duration / duration:.1f}x)"
            if audio_duration > 0
            else ""
        )

        return TranscriptResult(
            audio_path=str(audio_path),
            sample_rate=file_sample_rate,
            language="en",  # vosk models are language-specific
            segments=segments_list,
            duration=audio_duration,
            stt={
                "engine": "vosk",
                "model_path": str(self.model_path),
                "diarization": self.enable_diarization,
            },
            speakers=speakers,
            diarization_enabled=self.enable_diarization,
        )

    def unload_model(self) -> None:
        """unload Vosk models and cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.spk_model is not None:
            del self.spk_model
            self.spk_model = None

        # Force garbage collection
        import gc

        gc.collect()

        console.print("[dim]vosk model unloaded[/dim]")


def save_transcript(result: TranscriptResult, base_path: Path) -> tuple[Path, Path]:
    """
    save transcript to text and json files.

    args:
        result: transcription result
        base_path: base path without extension

    returns:
        tuple of (text_path, json_path)
    """
    # save plain text with speaker labels if available
    text_path = base_path.with_suffix(".transcript.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for segment in result.segments:
            if segment.speaker_id is not None:
                f.write(f"[Speaker {segment.speaker_id}]: {segment.text}\n")
            else:
                f.write(f"{segment.text}\n")

    # save json with timestamps and speaker info
    json_path = base_path.with_suffix(".transcript.json")
    json_data = {
        "audio_path": result.audio_path,
        "sample_rate": result.sample_rate,
        "language": result.language,
        "segments": [asdict(s) for s in result.segments],
        "duration": result.duration,
        "stt": result.stt,
    }

    # add speaker info if available
    if result.speakers:
        json_data["speakers"] = result.speakers
    if result.diarization_enabled:
        json_data["diarization_enabled"] = result.diarization_enabled

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    console.print("[green]✓[/green] transcript saved:")
    console.print(f"  text: {text_path}")
    console.print(f"  json: {json_path}")

    return text_path, json_path
