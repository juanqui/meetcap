"""model download utilities for meetcap"""

import os
from pathlib import Path
from typing import Optional
import urllib.request

from rich.console import Console
from rich.progress import Progress, DownloadColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


def ensure_whisper_model(
    model_name: str = "large-v3",
    models_dir: Optional[Path] = None,
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
        from faster_whisper import WhisperModel
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
    models_dir: Optional[Path] = None,
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
    try:
        import faster_whisper
    except ImportError:
        console.print("[yellow]⚠[/yellow] faster-whisper not installed")
        return False
    
    # try to load model to verify it works
    try:
        from faster_whisper import WhisperModel
        
        console.print(f"[cyan]verifying whisper model '{model_name}'...[/cyan]")
        
        # try loading with download enabled
        model = WhisperModel(
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
    models_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Optional[Path]:
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
            f"[green]✓[/green] GGUF model already exists: {model_path.name} "
            f"({file_size_mb:.1f} MB)"
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
                    f"downloading {model_name}",
                    total=total_size if total_size > 0 else None
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
    models_dir: Optional[Path] = None,
    force_download: bool = False,
    model_choice: str = "thinking",
) -> Optional[Path]:
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


def verify_qwen_model(
    models_dir: Optional[Path] = None,
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