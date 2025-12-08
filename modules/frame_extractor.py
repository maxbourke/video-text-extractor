#!/usr/bin/env -S uv run
"""
Frame Extraction Module

Extracts frames from video files using FFmpeg at specified intervals.
Can be run standalone or imported by other modules.

Usage:
    uv run frame_extractor.py --video path/to/video.mp4 --fps 1 --output frames/
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pathlib",
# ]
# ///

import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os


class FrameExtractor:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get video info: {e}")
    
    def extract_frames(self, video_path: str, fps: float = 1.0, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to input video
            fps: Frames per second to extract (default 1.0 = 1 frame per second)
            output_dir: Directory to save frames (optional)
            
        Returns:
            List of extracted frame file paths
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use provided output_dir, instance output_dir, or create temp directory
        if output_dir:
            frames_dir = Path(output_dir)
        elif self.output_dir:
            frames_dir = self.output_dir
        else:
            frames_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info for duration calculation
        video_info = self.get_video_info(video_path)
        duration = float(video_info['format']['duration'])
        
        # Output pattern for frames
        output_pattern = frames_dir / f"{video_path.stem}_frame_%04d.png"
        
        # FFmpeg command to extract frames
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps}',
            '-y',  # Overwrite output files
            str(output_pattern)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find all extracted frames
            frame_files = sorted(frames_dir.glob(f"{video_path.stem}_frame_*.png"))
            frame_paths = [str(f) for f in frame_files]
            
            print(f"Extracted {len(frame_paths)} frames from {video_path.name}")
            print(f"Duration: {duration:.1f}s, FPS: {fps}, Expected frames: {int(duration * fps)}")
            
            return frame_paths
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    
    def extract_frame_with_timestamp(self, video_path: str, timestamp: float, output_path: Optional[str] = None) -> str:
        """Extract a single frame at specific timestamp"""
        video_path = Path(video_path)
        
        if output_path:
            output_file = Path(output_path)
        else:
            output_dir = self.output_dir or Path(tempfile.mkdtemp(prefix="video_frames_"))
            output_file = output_dir / f"{video_path.stem}_t{timestamp:.1f}s.png"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-ss', str(timestamp),
            '-vframes', '1',
            '-y',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return str(output_file)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract frame at {timestamp}s: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--output", help="Output directory for frames")
    parser.add_argument("--timestamp", type=float, help="Extract single frame at specific timestamp")
    
    args = parser.parse_args()
    
    extractor = FrameExtractor(args.output)
    
    if args.timestamp is not None:
        frame_path = extractor.extract_frame_with_timestamp(args.video, args.timestamp)
        print(f"Extracted frame: {frame_path}")
    else:
        frame_paths = extractor.extract_frames(args.video, args.fps, args.output)
        for path in frame_paths:
            print(path)


if __name__ == "__main__":
    main()