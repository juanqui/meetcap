# Development Guide for meetcap

This project uses [Hatch](https://hatch.pypa.io/latest/) for dependency management, environment handling, and build processes. This guide will help you get started with development and releasing new versions.

## Prerequisites

- macOS (required for audio capture functionality)
- Python 3.10 or higher
- [Hatch](https://hatch.pypa.io/latest/install/) installed globally
- ffmpeg (install with `brew install ffmpeg`)

### Installing Hatch

```bash
# Install via pipx (recommended)
pipx install hatch

# Or via pip
pip install hatch

# Or via conda
conda install -c conda-forge hatch
```

## Project Overview

meetcap is an offline meeting recorder & summarizer for macOS that captures system audio and microphone simultaneously, transcribes locally with Whisper, and summarizes with a local LLM.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/juanqui/meetcap.git
   cd meetcap
   ```

2. **Create and activate the default development environment**
   ```bash
   # Hatch will automatically create a virtual environment and install dependencies
   hatch shell
   ```

3. **Install development dependencies**
   ```bash
   # Install with development dependencies
   pip install -e .[dev]
   ```

4. **Verify the installation**
   ```bash
   # Run tests to ensure everything is working
   hatch run test
   ```

## Environment Management

### Default Development Environment

The project includes a pre-configured default environment with all essential development dependencies:

```bash
# Enter the development shell
hatch shell

# Run commands in the environment without entering the shell
hatch run <command>

# Show available environments
hatch env show
```

### Working with Optional Dependencies

The project includes optional dependency groups for different functionality:

- **`stt`**: Speech-to-text with faster-whisper
- **`llm`**: Local LLM support with llama-cpp-python
- **`dev`**: Development tools and testing dependencies

#### Installing Optional Dependencies

```bash
# Install the project in editable mode with specific extras
pip install -e ".[stt]"

# Install multiple groups
pip install -e ".[stt,llm]"

# For complete development setup
pip install -e ".[dev,stt,llm]"
```

## Development Tasks

All common development tasks are configured as Hatch scripts:

### Testing

```bash
# Run all tests
hatch run test

# Run tests with coverage reporting
hatch run test-cov

# Run specific test files or patterns
hatch run test tests/test_cli.py
hatch run test -k "test_devices"
hatch run test -v --tb=short
```

### Code Quality

```bash
# Format code with ruff
hatch run format

# Run linters and format checks
hatch run lint

# Run individual tools manually if needed
hatch run ruff format --check .
hatch run ruff check .
hatch run mypy meetcap
```

### Running the CLI

```bash
# Run meetcap commands in development
hatch run record           # Record from audio device
hatch run devices          # List audio devices
hatch run verify           # Quick verification

# Or use the full CLI
hatch run python -m meetcap.cli --help
```

## Development Workflow

### 1. Setting Up for Development

```bash
# Clone and enter the project
git clone https://github.com/juanqui/meetcap.git
cd meetcap

# Set up development environment
hatch shell

# Install with development dependencies
pip install -e .[dev,stt,llm]

# Install pre-commit hooks (see Git Hooks section)
pre-commit install
```

### 2. Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature

# Make your changes...

# Format and lint your code
hatch run format
hatch run lint

# Run tests to ensure nothing is broken
hatch run test
```

### 3. Before Committing

```bash
# Run full test suite with coverage
hatch run test-cov

# Ensure code is properly formatted
hatch run lint

# Check that build works
hatch build
```

## Releasing New Versions

This project uses `bump2version` for version management and automated GitHub Actions for releases.

### Version Bumping

```bash
# Install bump2version if not already available
pip install bump2version

# Bump patch version (e.g., 1.0.0 → 1.0.1)
bump2version patch

# Bump minor version (e.g., 1.0.0 → 1.1.0)
bump2version minor

# Bump major version (e.g., 1.0.0 → 2.0.0)
bump2version major
```

This will:
1. Update the version in `meetcap/__init__.py`
2. Create a git commit with the version change
3. Create a git tag with the new version (e.g., `v1.0.1`)

### Publishing a Release

```bash
# After bumping version, push the tag to trigger release
git push origin main --tags
```

This will automatically:
1. Run CI tests
2. Build the package
3. Create a GitHub release with release notes
4. Publish to PyPI (requires `PYPI_API_TOKEN` secret to be configured)

### Manual Release (if needed)

```bash
# Build the package locally
hatch build

# Check the build
pip install twine
twine check dist/*

# Upload to PyPI (requires API token)
twine upload dist/*
```

## Git Hooks and Pre-commit

The project uses pre-commit hooks to ensure code quality:

### Setting Up Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

### Available Hooks

- **black**: Code formatting
- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **check-toml**: Validate TOML files
- **check-merge-conflict**: Check for merge conflict markers

## macOS-Specific Requirements

Since meetcap is designed for macOS audio capture:

### Required Permissions

1. **Microphone Access**: System Preferences → Privacy & Security → Microphone
2. **Input Monitoring**: System Preferences → Privacy & Security → Input Monitoring (for hotkeys)

### Audio Setup

For optimal recording:
1. Install BlackHole 2ch: `brew install blackhole-2ch`
2. Create Audio Aggregate Device in Audio MIDI Setup
3. Configure system audio routing

See the main README.md for detailed audio setup instructions.

## Configuration

### Environment Variables

Key environment variables for development:

- **`MEETCAP_DEVICE`**: Override default audio device
- **`MEETCAP_STT_MODEL`**: Override Whisper model
- **`MEETCAP_LLM_MODEL`**: Override LLM model path
- **`MEETCAP_OUT_DIR`**: Override output directory
- **`LOG_LEVEL`**: Logging level (DEBUG, INFO, WARNING, ERROR)

Create a `.env` file in the project root:

```bash
# .env
LOG_LEVEL=DEBUG
MEETCAP_OUT_DIR=./recordings
```

### Tool Configuration

All tool configurations are in `pyproject.toml`:

- **Black**: 100 character line length
- **Ruff**: Comprehensive linting and formatting
- **mypy**: Type checking with meetcap-specific settings
- **pytest**: Test configuration with coverage
- **coverage**: Source tracking with HTML reports

## Python Version Compatibility

The project supports Python 3.10 through 3.12. To test against different Python versions:

```bash
# Create environments for different Python versions
hatch env create py310 --python=3.10
hatch env create py311 --python=3.11
hatch env create py312 --python=3.12

# Test against specific version
hatch run --env py311 test
```

## Building and Distribution

```bash
# Build wheel and source distribution
hatch build

# Clean build artifacts
hatch clean

# Check build output
twine check dist/*
```

## GitHub Actions

The project includes automated CI/CD:

### CI Workflow (.github/workflows/ci.yml)

- Runs on: push to main/develop, PRs
- Tests against Python 3.10, 3.11, 3.12 on macOS
- Performs: linting, testing, coverage, build verification

### Release Workflow (.github/workflows/release.yml)

- Triggers on: version tags (v*.*.*)
- Performs: build, GitHub release creation, PyPI publication
- Requires: `PYPI_API_TOKEN` secret in repository settings

## Troubleshooting

### Common Issues

1. **Audio permissions**: Grant microphone and input monitoring permissions in System Preferences

2. **ffmpeg not found**: Install with `brew install ffmpeg`

3. **Import errors**: Ensure you've installed in editable mode: `pip install -e .`

4. **Model download failures**: Check internet connection and model URLs in `meetcap/services/model_download.py`

5. **Test failures**: Some tests require audio devices; run with `--no-audio` if needed

### Getting Help

```bash
# Show hatch help
hatch --help

# Show environment information
hatch env show

# Show project dependencies
hatch dep show requirements

# Debug environment issues
hatch shell --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the development workflow above
4. Ensure all tests pass and code is properly formatted
5. Submit a pull request with a clear description

## Additional Resources

- [Hatch Documentation](https://hatch.pypa.io/latest/)
- [bump2version Documentation](https://github.com/c4urself/bump2version)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Project Issues](https://github.com/juanqui/meetcap/issues)
