"""llm-based meeting summarization using qwen3 via llama.cpp"""

import re
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class SummarizationService:
    """generate meeting summaries using local llm"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 32768,
        n_threads: int = 6,
        n_gpu_layers: int = 35,
        n_batch: int = 1024,
        temperature: float = 0.4,
        max_tokens: int = 4096,  # increased for more detailed summaries
        seed: int | None = None,
    ):
        """
        initialize summarization service.

        args:
            model_path: path to qwen3 gguf model file
            n_ctx: context window size
            n_threads: number of cpu threads
            n_gpu_layers: layers to offload to gpu (metal)
            n_batch: batch size for prompt processing
            temperature: sampling temperature (0.2-0.6 recommended)
            max_tokens: max tokens to generate
            seed: random seed for reproducibility
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.llm = None

    def _load_model(self):
        """lazy load the llm model."""
        if self.llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python not installed. "
                "install with metal support: "
                "CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
            ) from e

        console.print(f"[cyan]loading llm model from {self.model_path.name}...[/cyan]")

        # initialize llama with metal support
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            seed=self.seed if self.seed is not None else -1,
            verbose=False,
        )

    def load_model(self) -> None:
        """explicitly load the llm model."""
        self._load_model()

    def unload_model(self) -> None:
        """unload llama-cpp-python model and cleanup resources."""
        if self.llm is not None:
            # llama-cpp-python handles cleanup in destructor
            del self.llm
            self.llm = None

            # Clear Metal/GPU cache if available
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except (ImportError, AttributeError):
                pass

            # Force garbage collection
            import gc

            gc.collect()

            console.print("[dim]llm model unloaded[/dim]")

    def is_loaded(self) -> bool:
        """check if model is currently loaded."""
        return self.llm is not None

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

    def summarize(
        self,
        transcript_text: str,
        meeting_title: str | None = None,
        attendees: list[str] | None = None,
        has_speaker_info: bool = False,
        manual_notes_path: Path | None = None,
    ) -> str:
        """
        generate meeting summary from transcript.

        args:
            transcript_text: full transcript text
            meeting_title: optional meeting title
            attendees: optional list of attendees
            has_speaker_info: whether transcript includes speaker labels

        returns:
            markdown-formatted summary
        """
        # load model if needed
        self._load_model()

        console.print("[cyan]generating meeting summary...[/cyan]")
        start_time = time.time()

        # prepare prompt for detailed summary
        if has_speaker_info:
            # enhanced prompt when speaker information is available
            system_prompt = (
                "you are an expert meeting note-taker who creates comprehensive, actionable meeting summaries. "
                "the transcript includes speaker labels (e.g., [Speaker 1], [Speaker 2]). "
                "analyze the transcript and produce detailed notes with these exact sections:\n\n"
                "## Meeting Title\n"
                "generate a concise title (2-4 words) that captures the main topic of the meeting.\n"
                "the title should be in PascalCase with no spaces (e.g., 'ProductRoadmap', 'TeamRetrospective').\n"
                "write only the title on a single line, nothing else in this section.\n\n"
                "## Participants\n"
                "list the speakers identified in the transcript:\n"
                "- note their key contributions or roles in the discussion\n"
                "- identify who led the meeting if apparent\n\n"
                "## Summary\n"
                "provide a comprehensive 3-5 paragraph summary covering:\n"
                "- main topics discussed and context\n"
                "- key points raised by specific speakers\n"
                "- important details, data, or examples mentioned\n"
                "- overall meeting outcome and next steps\n\n"
                "## Key Discussion Points\n"
                "list 5-10 bullet points of the most important topics discussed:\n"
                "- include specific details and context for each point\n"
                "- attribute key points to specific speakers when relevant\n"
                "- note any disagreements or alternative viewpoints between speakers\n"
                "- highlight critical information or insights shared\n\n"
                "## Decisions\n"
                "list all decisions made during the meeting:\n"
                "- be specific about what was decided\n"
                "- include rationale if discussed\n"
                "- note which speaker made or supported the decision\n"
                "- if no decisions were made, write 'no formal decisions made'\n\n"
                "## Action Items\n"
                "list all tasks and follow-ups mentioned:\n"
                "- format: - [ ] owner (Speaker X) — detailed task description (due: yyyy-mm-dd)\n"
                "- if no owner mentioned, use 'tbd' as owner\n"
                "- if no date mentioned, use 'tbd' for date\n"
                "- include context for why each action is needed\n"
                "- if no action items, write 'no action items identified'\n\n"
                "## Notable Quotes\n"
                "include 2-3 important verbatim quotes with speaker attribution\n\n"
                "be thorough and detailed while maintaining clarity. "
                "do not include any thinking tags or meta-commentary."
            )
        else:
            # standard prompt without speaker information
            system_prompt = (
                "you are an expert meeting note-taker who creates comprehensive, actionable meeting summaries. "
                "analyze the transcript and produce detailed notes with these exact sections:\n\n"
                "## Meeting Title\n"
                "generate a concise title (2-4 words) that captures the main topic of the meeting.\n"
                "the title should be in PascalCase with no spaces (e.g., 'ProductRoadmap', 'TeamRetrospective').\n"
                "write only the title on a single line, nothing else in this section.\n\n"
                "## Summary\n"
                "provide a comprehensive 3-5 paragraph summary covering:\n"
                "- main topics discussed and context\n"
                "- key points raised by participants\n"
                "- important details, data, or examples mentioned\n"
                "- overall meeting outcome and next steps\n\n"
                "## Key Discussion Points\n"
                "list 5-10 bullet points of the most important topics discussed:\n"
                "- include specific details and context for each point\n"
                "- note any disagreements or alternative viewpoints\n"
                "- highlight critical information or insights shared\n\n"
                "## Decisions\n"
                "list all decisions made during the meeting:\n"
                "- be specific about what was decided\n"
                "- include rationale if discussed\n"
                "- note who made or supported the decision\n"
                "- if no decisions were made, write 'no formal decisions made'\n\n"
                "## Action Items\n"
                "list all tasks and follow-ups mentioned:\n"
                "- format: - [ ] owner — detailed task description (due: yyyy-mm-dd)\n"
                "- if no owner mentioned, use 'tbd' as owner\n"
                "- if no date mentioned, use 'tbd' for date\n"
                "- include context for why each action is needed\n"
                "- if no action items, write 'no action items identified'\n\n"
                "## Notable Quotes\n"
                "include 2-3 important verbatim quotes that capture key insights or decisions\n\n"
                "be thorough and detailed while maintaining clarity. "
                "do not include any thinking tags or meta-commentary."
            )

        # read manual notes if available
        manual_notes_text = ""
        if manual_notes_path and manual_notes_path.exists():
            try:
                with open(manual_notes_path, encoding="utf-8") as f:
                    manual_notes_text = f.read()
                console.print("[dim]manual notes found, including in summary[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")

        # build user prompt with manual notes
        user_prompt_parts = []

        if meeting_title:
            user_prompt_parts.append(f"meeting: {meeting_title}")
        if attendees:
            user_prompt_parts.append(f"attendees: {', '.join(attendees)}")

        # add manual notes first if available
        if manual_notes_text:
            user_prompt_parts.append(f"manual notes:\n{manual_notes_text}")

        user_prompt_parts.append(f"transcript:\n{transcript_text}")
        user_prompt = "\n\n".join(user_prompt_parts)

        # chunk if needed
        summaries = []
        if len(user_prompt) > self.n_ctx * 3:  # rough estimate: ~4 chars per token
            chunks = self._chunk_transcript(transcript_text, chunk_size=self.n_ctx * 2)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"processing {len(chunks)} chunks...", total=len(chunks))

                for i, chunk in enumerate(chunks):
                    chunk_prompt = f"transcript chunk {i + 1}/{len(chunks)}:\n{chunk}"
                    summary = self._generate_summary(system_prompt, chunk_prompt)
                    summaries.append(summary)
                    progress.update(task, advance=1)

            # merge summaries
            if len(summaries) > 1:
                merge_prompt = (
                    "merge these partial summaries into one final summary:\n\n"
                    + "\n---\n".join(summaries)
                )
                final_summary = self._generate_summary(system_prompt, merge_prompt)
            else:
                final_summary = summaries[0]
        else:
            # single pass for short transcripts
            with Progress(
                SpinnerColumn(),
                TextColumn("generating summary..."),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("", total=None)
                final_summary = self._generate_summary(system_prompt, user_prompt)

        duration = time.time() - start_time
        console.print(f"[green]✓[/green] summary generated in {duration:.1f}s")

        return final_summary

    def _generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        """
        generate summary using the llm.

        args:
            system_prompt: system instructions
            user_prompt: user input with transcript

        returns:
            generated summary text
        """
        # format messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # generate response
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["<|endoftext|>", "<|im_end|>"],  # qwen stop tokens
        )

        raw_output = response["choices"][0]["message"]["content"].strip()

        # debug: log if thinking tags are detected
        if "<think" in raw_output.lower() or "</think" in raw_output.lower():
            console.print("[dim]detected thinking tags in output, cleaning...[/dim]")
            # optionally log first 200 chars for debugging
            # console.print(f"[dim]raw start: {raw_output[:200]}...[/dim]")

        cleaned_output = self._clean_thinking_tags(raw_output)

        # warn if output seems empty after cleaning
        if not cleaned_output or len(cleaned_output) < 10:
            console.print("[yellow]⚠ output seems very short after cleaning[/yellow]")

        return cleaned_output

    def _clean_thinking_tags(self, text: str) -> str:
        """
        remove thinking tags from llm output.

        qwen3-thinking models include <think>...</think> tags that should be removed.
        handles edge cases like malformed tags, missing opening tags, etc.

        args:
            text: raw llm output possibly containing thinking tags

        returns:
            cleaned text without thinking tags
        """
        # handle case where content appears before </think> without opening tag
        # this catches everything from start until </think> if no opening tag exists
        if "</think>" in text.lower() and "<think" not in text.lower():
            # find the position of </think> and remove everything before it
            pattern = r"(?is)^.*?</think\s*>"
            cleaned = re.sub(pattern, "", text)
        else:
            # standard removal of <think>...</think> with content
            # (?is) = case-insensitive and dot matches newlines
            # [^>]*? = allows attributes or malformed opening tags
            cleaned = re.sub(r"(?is)<think[^>]*?>.*?</think\s*>", "", text)

        # also handle <thinking>...</thinking> variant
        if "</thinking>" in cleaned.lower() and "<thinking" not in cleaned.lower():
            pattern = r"(?is)^.*?</thinking\s*>"
            cleaned = re.sub(pattern, "", cleaned)
        else:
            cleaned = re.sub(r"(?is)<thinking[^>]*?>.*?</thinking\s*>", "", cleaned)

        # remove any remaining lone tags (opening or closing)
        cleaned = re.sub(r"(?i)</?think[^>]*?>", "", cleaned)
        cleaned = re.sub(r"(?i)</?thinking[^>]*?>", "", cleaned)

        # clean up any extra whitespace that may be left
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse multiple newlines
        cleaned = re.sub(r"^\n+", "", cleaned)  # remove leading newlines

        return cleaned.strip()

    def _chunk_transcript(self, text: str, chunk_size: int) -> list[str]:
        """
        split transcript into chunks for processing.

        args:
            text: full transcript text
            chunk_size: approximate size of each chunk in chars

        returns:
            list of text chunks
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_size + word_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_len
            else:
                current_chunk.append(word)
                current_size += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


def save_summary(summary_text: str, base_path: Path) -> Path:
    """
    save summary to markdown file.

    args:
        summary_text: generated summary markdown
        base_path: base path without extension

    returns:
        path to saved summary file
    """
    summary_path = base_path.with_suffix(".summary.md")

    # ensure proper markdown formatting
    if not summary_text.startswith("## "):
        # add default structure if missing (shouldn't happen with new prompt)
        summary_text = (
            "## summary\n\n" + summary_text + "\n\n"
            "## key discussion points\n\n(none identified)\n\n"
            "## decisions\n\n(none identified)\n\n"
            "## action items\n\n(none identified)\n\n"
            "## notable quotes\n\n(none identified)"
        )

    # add metadata header
    from datetime import datetime

    header = (
        f"# Meeting Summary\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n"
    )

    final_content = header + summary_text

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    console.print(f"[green]✓[/green] summary saved: {summary_path}")

    return summary_path


def extract_meeting_title(summary_text: str, transcript_text: str = "") -> str:
    """
    extract meeting title from summary or generate from transcript.

    args:
        summary_text: generated summary with meeting title section
        transcript_text: optional transcript for fallback title generation

    returns:
        meeting title in PascalCase format (no spaces)
    """
    import re

    # try to extract from ## Meeting Title section
    title_match = re.search(r"## Meeting Title\s*\n([^\n]+)", summary_text, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        # remove any markdown formatting
        title = re.sub(r"[*_`'\"]", "", title)
        # if already in PascalCase with no spaces, return as-is
        if " " not in title and title[0].isupper():
            return title
        # otherwise ensure PascalCase (remove spaces and capitalize)
        title = "".join(word.capitalize() for word in title.split())
        if title and len(title) > 2:
            return title

    # fallback: try to extract from first few words of summary
    summary_match = re.search(r"## Summary\s*\n([^\n]+)", summary_text, re.IGNORECASE)
    if summary_match:
        first_line = summary_match.group(1).strip()
        # extract key words (nouns/verbs)
        words = re.findall(r"\b[A-Z][a-z]+\b", first_line)
        if len(words) >= 2:
            title = "".join(words[:3])  # take first 2-3 capitalized words
            if title and len(title) > 2:
                return title

    # last resort: generate from transcript keywords
    if transcript_text:
        # find most common meaningful words
        words = re.findall(r"\b[a-z]{4,}\b", transcript_text.lower())
        if words:
            from collections import Counter

            word_counts = Counter(words)
            # filter out common words
            common_words = {
                "that",
                "this",
                "with",
                "from",
                "have",
                "been",
                "will",
                "would",
                "could",
                "should",
            }
            filtered = [(w, c) for w, c in word_counts.most_common(10) if w not in common_words]
            if filtered:
                # take top 2 words and capitalize
                title_words = [word.capitalize() for word, _ in filtered[:2]]
                return "".join(title_words) + "Meeting"

    # absolute fallback
    return "UntitledMeeting"
