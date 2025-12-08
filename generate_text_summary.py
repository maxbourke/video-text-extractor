#!/usr/bin/env -S uv run
"""
Generate Human-Friendly Text Summary

Creates a readable text summary from video analysis JSON results.
Shows frame number, timestamp, and extracted text in a clean format.

Usage:
    uv run generate_text_summary.py --json results.json
    uv run generate_text_summary.py --json results.json --output summary.txt
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pathlib",
# ]
# ///

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def generate_text_summary(analysis_data: Dict[str, Any], include_metadata: bool = True) -> str:
    """
    Generate human-readable text summary from analysis JSON
    
    Args:
        analysis_data: Parsed JSON analysis results
        include_metadata: Whether to include video metadata header
        
    Returns:
        Formatted text summary string
    """
    lines = []
    
    if include_metadata:
        # Header with video info
        video_name = analysis_data.get('video_name', 'Unknown Video')
        duration = analysis_data.get('video_info', {}).get('duration', 0)
        total_frames = analysis_data.get('analysis_summary', {}).get('total_frames', 0)
        frames_with_text = analysis_data.get('analysis_summary', {}).get('frames_with_text', 0)
        coverage = analysis_data.get('analysis_summary', {}).get('text_coverage_percent', 0)
        
        lines.extend([
            f"Text Summary: {video_name}",
            f"Duration: {format_timestamp(duration)} | Frames: {total_frames} | Text Found: {frames_with_text} ({coverage}%)",
            "=" * 80,
            ""
        ])
    
    # Extract frames with text
    frames_with_text = analysis_data.get('frames_with_text', [])
    
    if not frames_with_text:
        lines.append("No text found in video.")
        return "\n".join(lines)
    
    # Process each frame with text
    for i, frame_data in enumerate(frames_with_text, 1):
        timestamp = frame_data.get('timestamp', 0)
        timestamp_str = format_timestamp(timestamp)
        extracted_text = frame_data.get('extracted_text', '').strip()
        
        # Format: frame_number. MM:SS:\ntext
        lines.append(f"{i}. {timestamp_str}:")
        
        if extracted_text:
            # Indent each line of extracted text
            text_lines = extracted_text.split('\n')
            for text_line in text_lines:
                if text_line.strip():
                    lines.append(f"   {text_line.strip()}")
        else:
            lines.append("   [Text detection failed or no readable text]")
        
        lines.append("")  # Empty line between frames
    
    return "\n".join(lines)


def generate_detailed_summary(analysis_data: Dict[str, Any]) -> str:
    """Generate more detailed summary with confidence scores and regions"""
    lines = []
    
    # Header
    video_name = analysis_data.get('video_name', 'Unknown Video')
    lines.extend([
        f"Detailed Text Analysis: {video_name}",
        "=" * 80,
        ""
    ])
    
    frames_with_text = analysis_data.get('frames_with_text', [])
    
    for i, frame_data in enumerate(frames_with_text, 1):
        timestamp = frame_data.get('timestamp', 0)
        timestamp_str = format_timestamp(timestamp)
        extracted_text = frame_data.get('extracted_text', '').strip()
        text_regions = frame_data.get('text_regions', 0)
        avg_confidence = frame_data.get('avg_confidence', 0)
        
        lines.extend([
            f"Frame {i} @ {timestamp_str} ({timestamp:.1f}s)",
            f"Regions: {text_regions} | Detection Confidence: {avg_confidence:.2f}",
            ""
        ])
        
        # OCR results if available
        ocr_data = frame_data.get('ocr', {})
        if ocr_data:
            regions_with_text = ocr_data.get('regions_with_text', 0)
            total_regions = ocr_data.get('total_regions', 0)
            lines.append(f"OCR Success: {regions_with_text}/{total_regions} regions")
            
            # Region details
            region_results = ocr_data.get('region_results', [])
            for j, region in enumerate(region_results):
                region_text = region.get('extracted_text', '').strip()
                ocr_confidence = region.get('ocr_confidence', 0)
                
                if region_text:
                    lines.extend([
                        f"  Region {j+1} (confidence: {ocr_confidence:.2f}):",
                        f"    \"{region_text}\""
                    ])
        
        if extracted_text:
            lines.extend([
                "Combined Text:",
                f"  \"{extracted_text}\"",
                ""
            ])
        else:
            lines.extend(["No readable text extracted", ""])
        
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate human-friendly text summary from video analysis JSON")
    parser.add_argument("--json", required=True, help="Path to analysis JSON file")
    parser.add_argument("--output", help="Output file path (default: print to stdout)")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed summary with confidence scores")
    parser.add_argument("--no-metadata", action="store_true", help="Skip video metadata header")
    
    args = parser.parse_args()
    
    # Load JSON data
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(json_path, 'r') as f:
            analysis_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate summary
    if args.detailed:
        summary = generate_detailed_summary(analysis_data)
    else:
        summary = generate_text_summary(analysis_data, include_metadata=not args.no_metadata)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(summary)
        
        print(f"Summary written to: {output_path}")
    else:
        print(summary)


if __name__ == "__main__":
    main()