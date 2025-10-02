# Fix: Opus Encoding Channel Layout Error

## Problem

When attempting to record with Opus format, ffmpeg was failing with the following error:

```
Invalid channel layout 2.1 for specified mapping family -1.
```

### Root Cause

The macOS AVFoundation audio input was providing audio in a `2.1` channel layout (2 front channels + 1 LFE channel), which is a surround sound configuration. The Opus encoder (`libopus`) doesn't support non-standard channel layouts like `2.1` without explicit channel mapping family configuration.

In the original code:
- For Opus and FLAC encoding, sample rate (`-ar`) and channel count (`-ac`) were **not** explicitly set
- ffmpeg was trying to encode whatever channel layout the input provided (2.1 in this case)
- Opus encoder rejected this unusual layout because it expects standard configurations (mono=1, stereo=2, 5.1=6, etc.)

## Solution

Added explicit audio resampling and channel conversion **before** codec encoding for both Opus and FLAC formats. This forces ffmpeg to convert any input channel layout to the standard stereo (2 channels) or mono (1 channel) format that the codecs expect.

### Changes Made

**File:** `meetcap/core/recorder.py`

**Modified:** `_build_ffmpeg_command()` method

#### Before:
```python
if audio_format == AudioFormat.OPUS:
    cmd.extend([
        "-c:a", "libopus",
        "-b:a", f"{opus_bitrate}k",
        "-vbr", "on",
        "-application", "voip",
        "-frame_duration", "20",
    ])
elif audio_format == AudioFormat.FLAC:
    cmd.extend([
        "-c:a", "flac",
        "-compression_level", str(flac_compression),
    ])
```

#### After:
```python
if audio_format == AudioFormat.OPUS:
    cmd.extend([
        "-ar", str(self.sample_rate),  # Set sample rate (48000)
        "-ac", str(self.channels),     # Force to stereo (2) or mono (1)
        "-c:a", "libopus",
        "-b:a", f"{opus_bitrate}k",
        "-vbr", "on",
        "-application", "voip",
        "-frame_duration", "20",
    ])
elif audio_format == AudioFormat.FLAC:
    cmd.extend([
        "-ar", str(self.sample_rate),  # Set sample rate
        "-ac", str(self.channels),     # Force to stereo or mono
        "-c:a", "flac",
        "-compression_level", str(flac_compression),
    ])
```

### Key Points

1. **`-ar` (audio rate):** Explicitly sets the sample rate to 48000 Hz (default)
2. **`-ac` (audio channels):** Forces conversion to 2 channels (stereo) regardless of input layout
3. **Order matters:** These flags must come **before** `-c:a` (codec specification) so ffmpeg applies the resampling/channel conversion before encoding

### Additional Cleanup

Removed redundant sample rate handling code that was only applied for dual input with non-WAV formats:

```python
# REMOVED: This is now redundant
if dual_input and audio_format != AudioFormat.WAV:
    cmd.extend(["-ar", str(self.sample_rate)])
```

## Testing

### Before Fix:
```bash
$ uv run meetcap record
# Error: Invalid channel layout 2.1 for specified mapping family -1.
```

### After Fix:
```bash
$ uv run meetcap record
✓ recording started: recording.opus
  device: BlackHole 2ch (index 1)
  format: OPUS @ 32 kbps
```

### Verification:
```bash
$ ffprobe recording.opus
# Should show: Stream #0:0: Audio: opus, 48000 Hz, stereo, fltp
```

## Impact

- **Fixes:** Opus and FLAC recording on systems where audio devices provide non-standard channel layouts
- **Compatibility:** Ensures recordings work consistently across different macOS audio devices
- **Quality:** No quality loss - we're already targeting stereo output, this just ensures it happens correctly
- **Performance:** Minimal impact - resampling from 2.1 to 2.0 is trivial

## Related Issues

- Original config migration issue: Config defaulting to Opus format
- CLI parameter override issue: CLI hardcoded defaults overriding config values

All three issues have now been resolved:
1. ✅ Config properly migrates and saves Opus format
2. ✅ CLI respects config values instead of hardcoded defaults
3. ✅ Opus encoding handles non-standard channel layouts

## Environment Details

- **OS:** macOS 14.x
- **Audio Device:** BlackHole 2ch (virtual audio device)
- **Input Layout:** 2.1 channels (2 front + 1 LFE)
- **Target Layout:** 2.0 channels (stereo)
- **Sample Rate:** 48000 Hz
- **ffmpeg version:** 7.x with libopus support
