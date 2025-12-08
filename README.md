# Video Text Extractor

Modular pipeline for detecting text presence in video frames. Uses lightweight models for efficient batch processing of videos.

## Quick Start

All scripts use uv with inline dependencies and include proper shebangs for direct execution:

```bash
# Test with a video file (includes OCR text extraction)
uv run video_text_pipeline.py --video "Test Data/your_video.mp4"
# OR execute directly:
./video_text_pipeline.py --video "Test Data/your_video.mp4"

# Quick sample analysis (5 frames)
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --quick-sample 5

# Custom frame rate and confidence
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --fps 0.5 --confidence 0.7

# Disable OCR (detection only)
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --no-ocr

# Fast OCR mode
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --ocr-level fast

# Generate human-friendly text summary
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --text-summary

# Generate detailed summary with confidence scores
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --detailed-summary

# Export frames with detected text to timestamped folder
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --export-frames

# Keep original frames AND export detected text frames
uv run video_text_pipeline.py --video "Test Data/your_video.mp4" --export-frames --keep-frames
```

## Architecture

The pipeline consists of modular components that can be used independently:

### 1. Frame Extraction (`modules/frame_extractor.py`)
- Extracts frames from video using FFmpeg
- Configurable sampling rate (frames per second)
- Can extract single frames at specific timestamps

```bash
# Using uv run:
uv run modules/frame_extractor.py --video video.mp4 --fps 1 --output frames/
# OR execute directly:
./modules/frame_extractor.py --video video.mp4 --fps 1 --output frames/
```

### 2. Text Detection Modules

#### Apple Vision Framework (`modules/apple_vision_detector.py`)
- Uses macOS native Vision framework
- Hardware accelerated on Apple Silicon
- Fast and efficient for M1/M2 Macs

```bash
uv run modules/apple_vision_detector.py --image frame.png --summary
```

### 3. OCR Text Extraction (`modules/apple_vision_ocr.py`)
- Apple Vision Framework OCR for text recognition
- Accurate and fast text extraction from detected regions
- Configurable recognition levels (accurate/fast)
- Language correction and multi-language support

```bash
uv run modules/apple_vision_ocr.py --image frame.png
uv run modules/apple_vision_ocr.py --image frame.png --crop 100,100,200,200
```

### 4. Text Summary Generator (`generate_text_summary.py`)
- Converts JSON analysis results to human-readable format
- Clean "frame_number. MM:SS: \n text" format
- Detailed mode with confidence scores and region analysis

```bash
uv run generate_text_summary.py --json results.json
uv run generate_text_summary.py --json results.json --detailed
```

### 5. Frame Exporter (`modules/frame_exporter.py`)
- Exports frames with detected text to organized timestamped folders
- Sequential naming with shortened filenames
- Includes metadata and summary files

```bash
uv run modules/frame_exporter.py --json results.json
uv run modules/frame_exporter.py --json results.json --output export_dir
```

#### DBNet-tiny (Coming Soon)
- Lightweight neural network for text detection
- Cross-platform compatibility
- Optimised for batch processing

### 6. Pipeline Orchestrator (`video_text_pipeline.py`)
- Coordinates frame extraction, text detection, and OCR
- Aggregates results with timestamps and extracted text
- Modular detector selection
- Configurable OCR settings
- Optional human-friendly text summary generation
- Integrated frame export for text-containing frames

## Available Text Detectors

- **apple_vision**: Apple's Vision framework (default, best for Mac)
- **dbnet_tiny**: Lightweight neural network (cross-platform, requires PyTorch)

To use DBNet:
```bash
uv run video_text_pipeline.py --video video.mp4 --detector dbnet_tiny
```

## Output Formats

### JSON Results (Default)
Detailed machine-readable analysis results:

```json
{
  "video_name": "sample_video.mp4",
  "analysis_summary": {
    "total_frames": 120,
    "frames_with_text": 45,
    "text_coverage_percent": 37.5,
    "processing_time_seconds": 8.2,
    "frames_per_second": 14.6
  },
  "frames_with_text": [
    {
      "timestamp": 15.0,
      "timestamp_str": "00:15",
      "has_text": true,
      "text_regions": 2,
      "extracted_text": "Lee Cooper\nLecturer\nUNSW Centre for Social Impact",
      "regions": [...],
      "ocr": {
        "total_regions": 2,
        "regions_with_text": 2,
        "full_text": "Lee Cooper\nLecturer\nUNSW Centre for Social Impact"
      }
    }
  ]
}
```

### Human-Friendly Text Summary (Optional)
Clean, readable format when using `--text-summary`:

```
Text Summary: Lee COMM5706 (Design for Social Innovation) FC01.mp4
Duration: 03:24 | Frames: 10 | Text Found: 2 (20.0%)
================================================================================

1. 00:20:
   Lee Cooper
   Lecturer
   UNSW Centre for Social Impact

2. 01:42:
   [Text detection failed or no readable text]
```

### Frame Export (Optional)
When using `--export-frames`, creates a timestamped folder with sequential frame images:

```
2025-12-02-T230103 text_detected_frames Lee_COMM5706_Design_for_Social_Innovation_FC01/
├── Lee_COMM5706_Design_for_Social_0001.png  # Frame at 00:00
├── Lee_COMM5706_Design_for_Social_0002.png  # Frame at 00:16  
├── Lee_COMM5706_Design_for_Social_0003.png  # Frame at 00:20
├── ...
├── frame_export_metadata.json              # Detailed metadata
└── frame_summary.txt                       # Human-readable summary
```

## Adding New Detectors

To add a new text detection module:

1. Create detector class following the interface:
```python
class YourDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        pass
    
    def get_text_summary(self, image_path: str) -> Dict[str, Any]:
        return {
            'has_text': bool,
            'text_regions': int,
            'regions': [...],
            'detector': 'your_detector'
        }
```

2. Add to `video_text_pipeline.py`:
```python
AVAILABLE_DETECTORS = {
    'apple_vision': AppleVisionTextDetector,
    'your_detector': YourDetector,
}
```

## Requirements

- **macOS**: Required for Apple Vision Framework
- **FFmpeg**: For video processing (`brew install ffmpeg`)
- **Python 3.8+**: With uv package manager

Dependencies are managed via inline script metadata, automatically installed by `uv run`.

## Performance

On MacBook Air M1:
- **Apple Vision**: ~2.3 fps processing rate (with OCR)
- **Frame extraction**: ~50 fps  
- **Memory usage**: <100MB for typical videos

Recent test results:
- 3:24 video at 0.5 fps: 17/102 frames with text (16.7% coverage), 44.4s processing
- 3:24 video at 0.25 fps: 7/51 frames with text (13.7% coverage), 22.5s processing