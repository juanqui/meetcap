"""model download utilities for meetcap"""

import urllib.request
import zipfile
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()


def ensure_whisper_model(
    model_name: str = "large-v3",
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    ensure whisper model is available, downloading if necessary.

    args:
        model_name: name of whisper model (e.g., 'large-v3')
        models_dir: directory to store models (default: ~/.meetcap/models)
        force_download: force re-download even if exists

    returns:
        path to model directory
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models"

    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    # check if model already downloaded
    model_path = models_dir / f"whisper-{model_name}"

    if model_path.exists() and not force_download:
        console.print(f"[green]✓[/green] whisper model already exists: {model_path}")
        return model_path

    # download model using faster-whisper
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        console.print(
            "[red]error:[/red] faster-whisper not installed\n"
            "[yellow]install with:[/yellow] pip install faster-whisper"
        )
        return None

    console.print(f"[cyan]downloading whisper model '{model_name}'...[/cyan]")
    console.print("[dim]this may take several minutes on first run[/dim]")

    try:
        # use faster-whisper's download functionality
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"downloading {model_name}...", total=None)

            # download model to cache
            model_path = download_model(
                model_name,
                cache_dir=str(models_dir),
                local_files_only=False,
            )

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] model downloaded to: {model_path}")
        return Path(model_path)

    except Exception as e:
        console.print(f"[red]error downloading model:[/red] {e}")
        return None


def verify_whisper_model(
    model_name: str = "large-v3",
    models_dir: Path | None = None,
) -> bool:
    """
    verify whisper model is available for use.

    args:
        model_name: name of whisper model
        models_dir: directory where models are stored

    returns:
        true if model is available
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models"

    models_dir = models_dir.expanduser()

    # check if faster-whisper is installed
    import importlib.util

    if importlib.util.find_spec("faster_whisper") is None:
        console.print("[yellow]⚠[/yellow] faster-whisper not installed")
        return False

    # try to load model to verify it works
    try:
        from faster_whisper import WhisperModel

        console.print(f"[cyan]verifying whisper model '{model_name}'...[/cyan]")

        # try loading with download enabled
        WhisperModel(
            model_name,
            device="cpu",  # use cpu for verification
            compute_type="int8",
            download_root=str(models_dir),
            local_files_only=False,
        )

        console.print(f"[green]✓[/green] whisper model '{model_name}' is ready")
        return True

    except Exception as e:
        console.print(f"[red]✗[/red] model verification failed: {e}")
        return False


def ensure_mlx_llm_model(
    model_name: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
) -> bool:
    """
    ensure mlx llm model is available, downloading if necessary via huggingface_hub.

    args:
        model_name: huggingface repo id for the mlx model

    returns:
        True if model is available, False if download failed
    """
    try:
        from huggingface_hub import snapshot_download

        console.print(f"[cyan]ensuring mlx llm model '{model_name}' is available...[/cyan]")
        snapshot_download(model_name)
        console.print(f"[green]✓[/green] mlx llm model ready: {model_name}")
        return True
    except Exception as e:
        console.print(f"[red]error downloading mlx llm model:[/red] {e}")
        return False


def ensure_mlx_whisper_model(
    model_name: str = "mlx-community/whisper-large-v3-turbo",
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    ensure mlx-whisper model is available, downloading if necessary.

    args:
        model_name: hugging face model name (e.g., 'mlx-community/whisper-large-v3-turbo')
        models_dir: directory to store models (default: ~/.meetcap/models/mlx-whisper)
        force_download: force re-download even if exists

    returns:
        path to model directory
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "mlx-whisper"

    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    # create model directory name from HF repo
    model_dir_name = model_name.replace("/", "--")
    model_path = models_dir / model_dir_name

    if model_path.exists() and not force_download:
        console.print(f"[green]✓[/green] mlx-whisper model already exists: {model_path}")
        return model_path

    # download model using mlx-whisper
    try:
        import mlx_whisper
    except ImportError:
        console.print(
            "[red]error:[/red] mlx-whisper not installed\n"
            "[yellow]install with:[/yellow] pip install mlx-whisper"
        )
        return None

    console.print(f"[cyan]downloading mlx-whisper model '{model_name}'...[/cyan]")
    console.print("[dim]this may take several minutes on first run[/dim]")

    try:
        # use mlx-whisper's download functionality
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"downloading {model_name}...", total=None)

            # create a small test file to trigger model download
            import math
            import tempfile
            import wave

            # create a 1-second test audio file
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sample_rate = 16000
                    duration = 1.0
                    frames = int(sample_rate * duration)

                    # generate simple sine wave without numpy
                    audio_data = []
                    for i in range(frames):
                        t = i / sample_rate
                        sample = math.sin(440 * 2 * math.pi * t) * 0.1
                        audio_int16 = int(sample * 32767)
                        # clamp to int16 range
                        audio_int16 = max(-32768, min(32767, audio_int16))
                        audio_data.append(audio_int16)

                    # write wav file
                    with wave.open(tmp.name, "w") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        # convert to bytes
                        import struct

                        audio_bytes = struct.pack("<" + "h" * len(audio_data), *audio_data)
                        wav_file.writeframes(audio_bytes)

                    # trigger model download by transcribing test audio
                    mlx_whisper.transcribe(tmp.name, path_or_hf_repo=model_name)
            finally:
                # clean up temp file
                if tmp_path:
                    try:
                        Path(tmp_path).unlink()
                    except OSError:
                        pass

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] mlx-whisper model downloaded: {model_name}")
        return model_path

    except Exception as e:
        console.print(f"[red]error downloading mlx-whisper model:[/red] {e}")
        return None


def verify_mlx_whisper_model(
    model_name: str = "mlx-community/whisper-large-v3-turbo",
    models_dir: Path | None = None,
) -> bool:
    """
    verify mlx-whisper model is available for use.

    args:
        model_name: hugging face model name
        models_dir: directory where models are stored

    returns:
        true if model is available
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "mlx-whisper"

    models_dir = models_dir.expanduser()

    # check if mlx-whisper is installed
    import importlib.util

    if importlib.util.find_spec("mlx_whisper") is None:
        console.print("[yellow]⚠[/yellow] mlx-whisper not installed")
        return False

    # check if running on apple silicon
    import platform

    if platform.processor() != "arm":
        console.print("[yellow]⚠[/yellow] mlx-whisper requires Apple Silicon (M1/M2/M3)")
        return False

    # try to load model to verify it works
    try:
        import mlx_whisper

        console.print(f"[cyan]verifying mlx-whisper model '{model_name}'...[/cyan]")

        # try loading model by creating a minimal test transcription
        import math
        import tempfile
        import wave

        # create a minimal test audio file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sample_rate = 16000
                duration = 0.1  # very short for verification
                frames = int(sample_rate * duration)

                # generate simple sine wave without numpy
                audio_data = []
                for i in range(frames):
                    t = i / sample_rate
                    sample = math.sin(440 * 2 * math.pi * t) * 0.1
                    audio_int16 = int(sample * 32767)
                    # clamp to int16 range
                    audio_int16 = max(-32768, min(32767, audio_int16))
                    audio_data.append(audio_int16)

                # write wav file
                with wave.open(tmp.name, "w") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    # convert to bytes
                    import struct

                    audio_bytes = struct.pack("<" + "h" * len(audio_data), *audio_data)
                    wav_file.writeframes(audio_bytes)

                # try to transcribe with the model
                mlx_whisper.transcribe(tmp.name, path_or_hf_repo=model_name)
        finally:
            # clean up temp file
            if tmp_path:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass

        console.print(f"[green]✓[/green] mlx-whisper model '{model_name}' is ready")
        return True

    except Exception as e:
        console.print(f"[red]✗[/red] mlx-whisper model verification failed: {e}")
        return False


def verify_mlx_llm_model(
    model_name: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
) -> bool:
    """
    verify mlx llm model is available for use.

    args:
        model_name: huggingface repo id for the mlx model

    returns:
        true if model is available
    """
    import importlib.util
    import platform

    # check if running on apple silicon
    if platform.processor() != "arm":
        console.print("[yellow]⚠[/yellow] mlx-vlm requires Apple Silicon (M1/M2/M3/M4)")
        return False

    # check if mlx_vlm is installed
    if importlib.util.find_spec("mlx_vlm") is None:
        console.print("[yellow]⚠[/yellow] mlx-vlm not installed")
        return False

    # check if model is cached locally
    try:
        from huggingface_hub import try_to_load_from_cache

        # check for config.json as a proxy for the full model
        result = try_to_load_from_cache(model_name, "config.json")
        if result is not None and not isinstance(result, type(None)):
            console.print(f"[green]✓[/green] mlx llm model ready: {model_name}")
            return True
        else:
            console.print(f"[yellow]⚠[/yellow] mlx llm model not cached: {model_name}")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] mlx llm model verification failed: {e}")
        return False


def ensure_vosk_model(
    model_name: str = "vosk-model-en-us-0.22",
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path | None:
    """
    ensure vosk model is available, downloading if necessary.

    args:
        model_name: name of vosk model (e.g., 'vosk-model-en-us-0.22')
        models_dir: directory to store models (default: ~/.meetcap/models/vosk)
        force_download: force re-download even if exists

    returns:
        path to model directory or None if failed
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "vosk"

    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    # vosk model URLs
    model_urls = {
        "vosk-model-small-en-us-0.15": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "size": "~507MB",
        },
        "vosk-model-en-us-0.22": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            "size": "~1.8GB",
        },
        "vosk-model-en-us-0.42-gigaspeech": {
            "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
            "size": "~3.3GB",
        },
    }

    if model_name not in model_urls:
        console.print(f"[red]error:[/red] unknown vosk model: {model_name}")
        console.print(f"[yellow]available models:[/yellow] {', '.join(model_urls.keys())}")
        return None

    model_info = model_urls[model_name]
    model_path = models_dir / model_name

    # check if model already exists
    if model_path.exists() and not force_download:
        # verify it has required files
        if (model_path / "conf" / "model.conf").exists():
            console.print(f"[green]✓[/green] vosk model already exists: {model_path}")
            return model_path

    console.print(f"[cyan]downloading vosk model '{model_name}'...[/cyan]")
    console.print(f"[dim]size: {model_info['size']}[/dim]")
    console.print("[yellow]this may take several minutes depending on your connection[/yellow]")

    try:
        # download zip file
        zip_path = models_dir / f"{model_name}.zip"
        temp_zip = zip_path.with_suffix(".tmp")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            with urllib.request.urlopen(model_info["url"]) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                task = progress.add_task(
                    f"downloading {model_name}", total=total_size if total_size > 0 else None
                )

                chunk_size = 8192 * 16  # 128KB chunks
                downloaded = 0

                with open(temp_zip, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task, completed=downloaded)

        # rename temp file
        temp_zip.rename(zip_path)

        console.print("[cyan]extracting model archive...[/cyan]")

        # extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(models_dir)

        # remove zip file to save space
        zip_path.unlink()

        # verify extraction
        if model_path.exists() and (model_path / "conf" / "model.conf").exists():
            console.print(f"[green]✓[/green] vosk model ready: {model_path}")
            return model_path
        else:
            console.print("[red]error:[/red] model extraction failed - missing required files")
            return None

    except Exception as e:
        console.print(f"[red]error downloading vosk model:[/red] {e}")
        # cleanup partial downloads
        if temp_zip.exists():
            temp_zip.unlink()
        if zip_path.exists():
            zip_path.unlink()
        return None


def ensure_vosk_spk_model(
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path | None:
    """
    ensure vosk speaker model is available, downloading if necessary.

    args:
        models_dir: directory to store models (default: ~/.meetcap/models/vosk)
        force_download: force re-download even if exists

    returns:
        path to speaker model directory or None if failed
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "vosk"

    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = "vosk-model-spk-0.4"
    model_url = "https://alphacephei.com/vosk/models/vosk-model-spk-0.4.zip"
    model_path = models_dir / model_name

    # check if model already exists
    if model_path.exists() and not force_download:
        # verify it has required files
        if (model_path / "model" / "final.raw").exists():
            console.print(f"[green]✓[/green] vosk speaker model already exists: {model_path}")
            return model_path

    console.print("[cyan]downloading vosk speaker model...[/cyan]")
    console.print("[dim]size: ~13MB[/dim]")

    try:
        # download zip file
        zip_path = models_dir / f"{model_name}.zip"
        temp_zip = zip_path.with_suffix(".tmp")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            with urllib.request.urlopen(model_url) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                task = progress.add_task(
                    "downloading speaker model", total=total_size if total_size > 0 else None
                )

                chunk_size = 8192 * 16  # 128KB chunks
                downloaded = 0

                with open(temp_zip, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task, completed=downloaded)

        # rename temp file
        temp_zip.rename(zip_path)

        console.print("[cyan]extracting speaker model archive...[/cyan]")

        # extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(models_dir)

        # remove zip file to save space
        zip_path.unlink()

        # verify extraction
        if model_path.exists() and (model_path / "model" / "final.raw").exists():
            console.print(f"[green]✓[/green] speaker model ready: {model_path}")
            return model_path
        else:
            console.print(
                "[red]error:[/red] speaker model extraction failed - missing required files"
            )
            return None

    except Exception as e:
        console.print(f"[red]error downloading speaker model:[/red] {e}")
        # cleanup partial downloads
        if temp_zip.exists():
            temp_zip.unlink()
        if zip_path.exists():
            zip_path.unlink()
        return None


def verify_vosk_model(
    model_name: str = "vosk-model-en-us-0.22",
    models_dir: Path | None = None,
) -> bool:
    """
    verify vosk model is available for use.

    args:
        model_name: name of vosk model
        models_dir: directory where models are stored

    returns:
        true if model is available
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "vosk"

    models_dir = models_dir.expanduser()

    # check if vosk is installed
    import importlib.util

    if importlib.util.find_spec("vosk") is None:
        console.print("[yellow]⚠[/yellow] vosk not installed")
        return False

    # check if model directory exists
    model_path = models_dir / model_name

    if not model_path.exists():
        console.print(f"[yellow]⚠[/yellow] vosk model not found: {model_name}")
        return False

    # check for required files
    required_files = [
        model_path / "conf" / "model.conf",
        model_path / "am" / "final.mdl",
    ]

    for file_path in required_files:
        if not file_path.exists():
            console.print(
                f"[yellow]⚠[/yellow] missing required file: {file_path.relative_to(model_path)}"
            )
            return False

    console.print(f"[green]✓[/green] vosk model '{model_name}' is ready")
    return True
