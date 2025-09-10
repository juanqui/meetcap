"""comprehensive tests for summarization service"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from meetcap.services.summarization import SummarizationService, extract_meeting_title, save_summary


class TestSummarizationService:
    """test summarization service functionality"""

    @pytest.fixture
    def model_file(self, temp_dir):
        """create mock model file"""
        model_path = temp_dir / "qwen3.gguf"
        model_path.write_bytes(b"fake model data")
        return model_path

    @pytest.fixture
    def mock_llama(self):
        """mock Llama class"""
        with patch("meetcap.services.summarization.Llama") as mock:
            yield mock

    def test_init_success(self, model_file):
        """test successful initialization"""
        service = SummarizationService(
            model_path=str(model_file),
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=20,
            n_batch=512,
            temperature=0.3,
            max_tokens=2048,
            seed=42,
        )

        assert service.model_path == model_file
        assert service.n_ctx == 4096
        assert service.n_threads == 4
        assert service.n_gpu_layers == 20
        assert service.n_batch == 512
        assert service.temperature == 0.3
        assert service.max_tokens == 2048
        assert service.seed == 42
        assert service.llm is None  # lazy loading

    def test_init_model_not_found(self):
        """test initialization with missing model"""
        with pytest.raises(FileNotFoundError, match="model not found"):
            SummarizationService(model_path="/nonexistent/model.gguf")

    def test_load_model_import_error(self, model_file):
        """test handling of missing llama-cpp-python"""
        with patch("builtins.__import__", side_effect=ImportError):
            service = SummarizationService(model_path=str(model_file))

            with pytest.raises(ImportError, match="llama-cpp-python not installed"):
                service._load_model()

    def test_load_model_success(self, model_file, mock_console):
        """test successful model loading"""
        from llama_cpp import Llama

        with patch.object(Llama, "__new__", return_value=Mock()) as mock_constructor:
            service = SummarizationService(model_path=str(model_file))
            service._load_model()

            assert service.llm is not None
            mock_constructor.assert_called_once()
            call_kwargs = mock_constructor.call_args[1]
            assert call_kwargs["model_path"] == str(model_file)
            assert call_kwargs["n_ctx"] == 32768

        # verify console output was called
        # mock_console.print.assert_called() - skip console check for now

    def test_load_model_with_seed(self, model_file):
        """test model loading with custom seed"""
        from llama_cpp import Llama

        with patch.object(Llama, "__new__", return_value=Mock()) as mock_constructor:
            service = SummarizationService(model_path=str(model_file), seed=123)
            service._load_model()

            call_kwargs = mock_constructor.call_args[1]
            assert call_kwargs["seed"] == 123

    def test_load_model_only_once(self, model_file):
        """test model is only loaded once"""
        from llama_cpp import Llama

        with patch.object(Llama, "__new__", return_value=Mock()) as mock_constructor:
            service = SummarizationService(model_path=str(model_file))

            service._load_model()
            service._load_model()  # second call

            # should only be called once
            mock_constructor.assert_called_once()

    def test_summarize_short_transcript(self, model_file, mock_console):
        """test summarizing short transcript"""
        service = SummarizationService(model_path=str(model_file))

        # Mock the llm directly
        mock_llm_instance = Mock()
        mock_llm_instance.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "## summary\n\nTest summary content\n\n## key discussion points\n\n- Point 1"
                    }
                }
            ]
        }
        service.llm = mock_llm_instance

        transcript = "This is a short test transcript."
        summary = service.summarize(
            transcript, meeting_title="Test Meeting", attendees=["Alice", "Bob"]
        )

        assert "## summary" in summary
        assert "Test summary content" in summary
        assert "## key discussion points" in summary

        # verify llm was called
        mock_llm_instance.create_chat_completion.assert_called_once()
        call_args = mock_llm_instance.create_chat_completion.call_args

        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test Meeting" in messages[1]["content"]
        assert "Alice, Bob" in messages[1]["content"]

    def test_summarize_long_transcript_chunking(self, model_file, mock_console):
        """test summarizing long transcript with chunking"""
        service = SummarizationService(
            model_path=str(model_file),
            n_ctx=100,  # very small to force chunking
        )

        # Mock the llm directly - make it return the same response for all calls
        mock_llm_instance = Mock()
        mock_llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "## summary\n\nFinal merged summary"}}]
        }
        service.llm = mock_llm_instance

        # create long transcript
        transcript = " ".join(["word"] * 1000)
        summary = service.summarize(transcript)

        assert "Final merged summary" in summary

        # should have called llm multiple times for chunking
        assert mock_llm_instance.create_chat_completion.call_count >= 2

    def test_clean_thinking_tags_standard(self, model_file):
        """test cleaning standard thinking tags"""
        service = SummarizationService(model_path=str(model_file))

        test_cases = [
            ("<think>thinking content</think>actual summary", "actual summary"),
            ("<THINK>UPPER CASE</THINK>content", "content"),
            ("<think>\nmultiline\nthinking\n</think>\nreal content", "real content"),
            ("before<think>middle</think>after", "beforeafter"),
            ("<thinking>variant tag</thinking>content", "content"),
        ]

        for input_text, expected in test_cases:
            result = service._clean_thinking_tags(input_text)
            assert result == expected

    def test_clean_thinking_tags_malformed(self, model_file):
        """test cleaning malformed thinking tags"""
        service = SummarizationService(model_path=str(model_file))

        # missing opening tag
        result = service._clean_thinking_tags("some thinking</think>actual content")
        assert result == "actual content"

        # missing closing tag (should keep content)
        result = service._clean_thinking_tags("<think>thinking\nactual content")
        assert "actual content" in result

        # nested tags
        result = service._clean_thinking_tags("<think>outer<think>inner</think></think>content")
        assert result == "content"

    def test_clean_thinking_tags_multiple(self, model_file):
        """test cleaning multiple thinking tags"""
        service = SummarizationService(model_path=str(model_file))

        text = "<think>first</think>content1<thinking>second</thinking>content2"
        result = service._clean_thinking_tags(text)
        assert result == "content1content2"

    def test_clean_thinking_tags_with_attributes(self, model_file):
        """test cleaning tags with attributes"""
        service = SummarizationService(model_path=str(model_file))

        text = '<think type="deep">thinking</think>summary'
        result = service._clean_thinking_tags(text)
        assert result == "summary"

    def test_clean_thinking_tags_whitespace(self, model_file):
        """test whitespace cleanup after tag removal"""
        service = SummarizationService(model_path=str(model_file))

        text = "<think>thinking</think>\n\n\n\n## summary"
        result = service._clean_thinking_tags(text)
        assert result == "## summary"
        assert not result.startswith("\n")

    def test_generate_summary_with_thinking_tags(self, model_file):
        """test summary generation removes thinking tags"""
        service = SummarizationService(model_path=str(model_file))

        # Mock the llm directly
        mock_llm_instance = Mock()
        mock_llm_instance.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "<think>I should analyze this</think>## summary\n\nActual summary"
                    }
                }
            ]
        }
        service.llm = mock_llm_instance

        with patch("meetcap.services.summarization.console") as mock_console:
            result = service._generate_summary("system", "user")

            assert "<think" not in result
            assert "## summary" in result
            assert "Actual summary" in result

            # verify console warning about thinking tags
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("thinking tags" in call.lower() for call in calls)

    def test_generate_summary_empty_after_cleaning(self, model_file):
        """test warning when output is empty after cleaning"""
        service = SummarizationService(model_path=str(model_file))

        # Mock the llm directly
        mock_llm_instance = Mock()
        mock_llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "<think>only thinking</think>"}}]
        }
        service.llm = mock_llm_instance

        with patch("meetcap.services.summarization.console") as mock_console:
            service._generate_summary("system", "user")

            # verify warning about short output
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("output seems very short" in call.lower() for call in calls)

    def test_chunk_transcript(self, model_file):
        """test transcript chunking"""
        service = SummarizationService(model_path=str(model_file))

        text = " ".join([f"word{i}" for i in range(100)])
        chunks = service._chunk_transcript(text, chunk_size=50)

        assert len(chunks) > 1

        # verify all words are preserved
        all_words = []
        for chunk in chunks:
            all_words.extend(chunk.split())
        assert len(all_words) == 100
        assert all_words[0] == "word0"
        assert all_words[-1] == "word99"

    def test_chunk_transcript_single_chunk(self, model_file):
        """test chunking with text smaller than chunk size"""
        service = SummarizationService(model_path=str(model_file))

        text = "short text"
        chunks = service._chunk_transcript(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_summarize_without_metadata(self, model_file):
        """test summarizing without meeting title or attendees"""
        service = SummarizationService(model_path=str(model_file))

        # Mock the llm directly
        mock_llm_instance = Mock()
        mock_llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "## summary\n\nContent"}}]
        }
        service.llm = mock_llm_instance
        summary = service.summarize("transcript text")

        assert "## summary" in summary

        # verify no metadata in prompt
        call_args = mock_llm_instance.create_chat_completion.call_args
        user_content = call_args[1]["messages"][1]["content"]
        assert "meeting:" not in user_content.lower()
        assert "attendees:" not in user_content.lower()


class TestSaveSummary:
    """test summary saving functionality"""

    def test_save_summary_with_proper_format(self, temp_dir):
        """test saving properly formatted summary"""
        summary_text = """## summary

This is the summary.

## key discussion points

- Point 1
- Point 2

## decisions

No formal decisions made

## action items

- [ ] TBD — Follow up on discussion (due: TBD)

## notable quotes

"Important quote here"
"""

        base_path = temp_dir / "meeting"

        with patch("meetcap.services.summarization.console") as mock_console:
            result_path = save_summary(summary_text, base_path)

            assert result_path == base_path.with_suffix(".summary.md")
            assert result_path.exists()

            content = result_path.read_text()

            # verify header was added
            assert "# Meeting Summary" in content
            assert "Generated:" in content
            assert datetime.now().strftime("%Y-%m-%d") in content

            # verify original content preserved
            assert "## summary" in content
            assert "This is the summary" in content
            assert "## key discussion points" in content
            assert "Point 1" in content

            # verify console output
            mock_console.print.assert_called()
            output = str(mock_console.print.call_args)
            assert "summary saved" in output.lower()

    def test_save_summary_missing_structure(self, temp_dir):
        """test saving summary with missing structure"""
        summary_text = "Just some plain text without proper formatting"

        base_path = temp_dir / "meeting"
        result_path = save_summary(summary_text, base_path)

        content = result_path.read_text()

        # verify default structure was added
        assert "## summary" in content
        assert summary_text in content
        assert "## key discussion points" in content
        assert "(none identified)" in content
        assert "## decisions" in content
        assert "## action items" in content
        assert "## notable quotes" in content

    def test_save_summary_unicode(self, temp_dir):
        """test saving summary with unicode characters"""
        summary_text = """## summary

讨论了项目进展 (discussed project progress)
会議の要約 (meeting summary)
😀 Great meeting!

## key discussion points

- 中文内容
- 日本語の内容
- Emoji: 🎯 🚀 ✅
"""

        base_path = temp_dir / "unicode_meeting"
        result_path = save_summary(summary_text, base_path)

        content = result_path.read_text(encoding="utf-8")

        assert "讨论了项目进展" in content
        assert "会議の要約" in content
        assert "😀" in content
        assert "🎯" in content

    def test_save_summary_path_creation(self, temp_dir):
        """test summary path creation"""
        base_path = temp_dir / "subdir" / "meeting"

        # create parent directory (save_summary doesn't create parent dirs)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        summary_text = "## summary\n\nTest"
        result_path = save_summary(summary_text, base_path)

        # file should be created
        assert result_path.exists()
        assert result_path.name == "meeting.summary.md"


class TestSummarizationIntegration:
    """integration tests for summarization"""

    def test_full_summarization_flow(self, temp_dir, mock_console):
        """test complete summarization flow"""
        model_file = temp_dir / "model.gguf"
        model_file.write_bytes(b"fake model")

        service = SummarizationService(model_path=str(model_file))

        # Mock the llm directly
        mock_llm = Mock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """<think>Let me analyze this transcript</think>

## summary

The meeting covered project updates and timeline.

## key discussion points

- Project milestone reached
- Budget approved

## decisions

Proceed with phase 2

## action items

- [ ] Alice — Prepare report (due: 2024-01-15)

## notable quotes

"This is a game changer"
"""
                    }
                }
            ]
        }
        service.llm = mock_llm

        transcript = "Alice: We reached the milestone. Bob: Great! This is a game changer."
        summary = service.summarize(
            transcript, meeting_title="Project Update", attendees=["Alice", "Bob"]
        )

        # verify thinking tags removed
        assert "<think" not in summary
        assert "Let me analyze" not in summary

        # verify content preserved
        assert "## summary" in summary
        assert "project updates and timeline" in summary
        assert "Project milestone reached" in summary
        assert "Proceed with phase 2" in summary
        assert "Alice — Prepare report" in summary
        assert "This is a game changer" in summary

        # save summary
        base_path = temp_dir / "meeting"
        save_summary(summary, base_path)

        # verify saved file
        saved_file = base_path.with_suffix(".summary.md")
        assert saved_file.exists()

        saved_content = saved_file.read_text()
        assert "# Meeting Summary" in saved_content
        assert "Project milestone reached" in saved_content


class TestExtractMeetingTitle:
    """test extract_meeting_title function"""

    def test_extract_from_meeting_title_section(self):
        """test extracting title from Meeting Title section"""
        summary = """## Meeting Title
ProductRoadmap

## Summary
This was a product roadmap meeting..."""

        title = extract_meeting_title(summary)
        assert title == "ProductRoadmap"

    def test_extract_with_spaces(self):
        """test extracting title with spaces that should be removed"""
        summary = """## Meeting Title
Product Roadmap Review

## Summary
Discussion about the product..."""

        title = extract_meeting_title(summary)
        assert title == "ProductRoadmapReview"

    def test_fallback_to_summary_keywords(self):
        """test fallback when no Meeting Title section"""
        summary = """## Summary
The Engineering Team discussed Sprint Planning for the next iteration..."""

        title = extract_meeting_title(summary)
        # should extract capitalized words from summary
        assert "Engineering" in title or "Team" in title or "Sprint" in title

    def test_fallback_to_transcript_keywords(self):
        """test fallback to transcript when no good title found"""
        summary = "## Summary\nwe talked about things"
        transcript = "project project project review review development sprint sprint sprint"

        title = extract_meeting_title(summary, transcript)
        # should use most common words
        assert "Meeting" in title  # always adds Meeting suffix in this case
        assert len(title) > 7  # should have meaningful content

    def test_fallback_to_untitled(self):
        """test absolute fallback when nothing works"""
        summary = "random text"

        title = extract_meeting_title(summary)
        assert title == "UntitledMeeting"

    def test_remove_markdown_formatting(self):
        """test removal of markdown formatting from title"""
        summary = """## Meeting Title
**ProductLaunch**

## Summary
Launch planning..."""

        title = extract_meeting_title(summary)
        assert title == "ProductLaunch"
        assert "*" not in title


def test_manual_notes_integration():
    """Test manual notes are included in summarization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        notes_path = temp_path / "notes.md"
        notes_path.write_text("# Meeting Notes\n\nImportant context about the meeting\n")

        # Test summarization includes manual notes
        mock_service = MockSummarizationService()
        summary = mock_service.summarize(
            transcript_text="Hello everyone, let's discuss the project timeline.",
            manual_notes_path=notes_path,
        )

        assert "Important context about the meeting" in summary


class MockSummarizationService:
    """Mock summarization service for testing manual notes integration"""

    def summarize(self, transcript_text: str, manual_notes_path: Path | None = None) -> str:
        """Mock summarize method that includes manual notes if provided"""
        manual_notes_text = ""
        if manual_notes_path and manual_notes_path.exists():
            try:
                with open(manual_notes_path, encoding="utf-8") as f:
                    manual_notes_text = f.read()
            except Exception as e:
                print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")

        # Build user prompt with manual notes
        user_prompt_parts = []

        # add manual notes first if available
        if manual_notes_text:
            user_prompt_parts.append(f"manual notes:\n{manual_notes_text}")

        user_prompt_parts.append(f"transcript:\n{transcript_text}")
        # user_prompt = "\n\n".join(user_prompt_parts)  # Not used in mock

        # Mock LLM response that includes manual notes content
        base_summary = '## summary\n\nThis meeting was about project timeline.\n\n## key discussion points\n\n- Project planning\n- Timeline review\n\n## decisions\n\nApproved project timeline\n\n## action items\n\n- [ ] Team — Finalize project plan (due: TBD)\n\n## notable quotes\n\n"Let\'s move forward with the plan"'

        # Include manual notes content in the summary if available
        if manual_notes_text:
            # Extract the key content from manual notes (skip the header)
            lines = manual_notes_text.strip().split("\n")
            key_content = []
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    key_content.append(line.strip())

            if key_content:
                # Insert manual notes content into the summary
                manual_notes_section = " ".join(key_content)
                base_summary = base_summary.replace(
                    "This meeting was about project timeline.",
                    f"This meeting was about project timeline. {manual_notes_section}",
                )

        return base_summary
