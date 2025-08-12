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


def download_gguf_model(
    model_url: str,
    model_name: str,
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path | None:
    """
    download a GGUF model from a direct URL.

    args:
        model_url: direct URL to the GGUF file
        model_name: name for the local file
        models_dir: directory to store models (default: ~/.meetcap/models)
        force_download: force re-download even if exists

    returns:
        path to downloaded model or None if failed
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models"

    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_name

    # check if already exists
    if model_path.exists() and not force_download:
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        console.print(
            f"[green]✓[/green] GGUF model already exists: {model_path.name} ({file_size_mb:.1f} MB)"
        )
        return model_path

    console.print(f"[cyan]downloading GGUF model '{model_name}'...[/cyan]")
    console.print(f"[dim]from: {model_url[:80]}...[/dim]")
    console.print("[yellow]this may take several minutes depending on your connection[/yellow]")

    try:
        # create a temporary file first
        temp_path = model_path.with_suffix(".tmp")

        # setup progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            # get file size first
            with urllib.request.urlopen(model_url) as response:
                total_size = int(response.headers.get("Content-Length", 0))

                if total_size == 0:
                    console.print("[yellow]warning: cannot determine file size[/yellow]")

                task = progress.add_task(
                    f"downloading {model_name}", total=total_size if total_size > 0 else None
                )

                # download in chunks
                chunk_size = 8192 * 16  # 128KB chunks
                downloaded = 0

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task, completed=downloaded)

        # move temp file to final location
        temp_path.rename(model_path)

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        console.print(
            f"[green]✓[/green] model downloaded: {model_path.name} ({file_size_mb:.1f} MB)"
        )
        return model_path

    except Exception as e:
        console.print(f"[red]error downloading model:[/red] {e}")
        # cleanup temp file if exists
        if temp_path.exists():
            temp_path.unlink()
        return None


def ensure_qwen_model(
    models_dir: Path | None = None,
    force_download: bool = False,
    model_choice: str = "thinking",
) -> Path | None:
    """
    ensure Qwen3-4B GGUF model is available, downloading if necessary.

    args:
        models_dir: directory to store models (default: ~/.meetcap/models)
        force_download: force re-download even if exists
        model_choice: which model variant ('thinking', 'instruct', 'gpt-oss')

    returns:
        path to model file or None if failed
    """
    # model choices
    models = {
        "thinking": {
            "url": "https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF/resolve/main/Qwen3-4B-Thinking-2507-UD-Q8_K_XL.gguf",
            "name": "Qwen3-4B-Thinking-2507-Q8_K_XL.gguf",
            "size": "~4-5GB",
        },
        "instruct": {
            "url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf",
            "name": "Qwen3-4B-Instruct-2507-Q8_K_XL.gguf",
            "size": "~4-5GB",
        },
        "gpt-oss": {
            "url": "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf",
            "name": "gpt-oss-20b-Q4_K_M.gguf",
            "size": "~11GB",
        },
    }

    if model_choice not in models:
        model_choice = "thinking"  # default

    model_info = models[model_choice]
    return download_gguf_model(model_info["url"], model_info["name"], models_dir, force_download)


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
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
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

        console.print(f"[green]✓[/green] mlx-whisper model '{model_name}' is ready")
        return True

    except Exception as e:
        console.print(f"[red]✗[/red] mlx-whisper model verification failed: {e}")
        return False


def verify_qwen_model(
    models_dir: Path | None = None,
) -> bool:
    """
    verify Qwen GGUF model is available for use.

    args:
        models_dir: directory where models are stored

    returns:
        true if model is available
    """
    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models"

    models_dir = models_dir.expanduser()
    model_path = models_dir / "Qwen3-4B-Thinking-2507-Q8_K_XL.gguf"

    if model_path.exists():
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        # GGUF files should be at least a few hundred MB
        if file_size_mb > 100:
            console.print(
                f"[green]✓[/green] Qwen model ready: {model_path.name} ({file_size_mb:.1f} MB)"
            )
            return True
        else:
            console.print(
                f"[yellow]⚠[/yellow] Qwen model file seems too small ({file_size_mb:.1f} MB)"
            )
            return False
    else:
        console.print("[yellow]⚠[/yellow] Qwen model not found")
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
