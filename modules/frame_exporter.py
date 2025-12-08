#!/usr/bin/env -S uv run
"""
Frame Exporter Module

Exports video frames with detected text to timestamped folders with sequential naming.
Designed to work with analysis results from the video text pipeline.

Usage:
    uv run frame_exporter.py --source frames_dir --output export_dir --results results.json
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pillow",
#     "pathlib",
# ]
# ///

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys


class FrameExporter:
    """Export frames with detected text to organized folders"""
    
    def __init__(self, base_output_dir: Optional[str] = None):
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path(".")
        
    def create_timestamped_folder(self, video_name: str, prefix: str = "text_detected_frames") -> Path:
        """
        Create timestamped folder for exported frames
        
        Args:
            video_name: Original video filename (for folder naming)
            prefix: Folder name prefix
            
        Returns:
            Path to created folder
        """
        timestamp = datetime.now().strftime("%Y-%m-%d-T%H%M%S")
        
        # Clean video name for folder naming
        clean_video_name = self.clean_filename(Path(video_name).stem)
        
        folder_name = f"{timestamp} {prefix} {clean_video_name}"
        folder_path = self.base_output_dir / folder_name
        
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    def clean_filename(self, filename: str, max_length: int = 50) -> str:
        """
        Clean and shorten filename for safe filesystem usage
        
        Args:
            filename: Original filename
            max_length: Maximum length for cleaned filename
            
        Returns:
            Cleaned filename safe for filesystem use
        """
        # Replace problematic characters
        clean = filename.replace(" ", "_").replace("(", "").replace(")", "")
        clean = "".join(c for c in clean if c.isalnum() or c in "._-")
        
        # Truncate if too long
        if len(clean) > max_length:
            clean = clean[:max_length]
        
        return clean
    
    def generate_frame_filename(self, original_video_name: str, frame_index: int, frame_extension: str = "png") -> str:
        """
        Generate sequential filename for exported frame
        
        Args:
            original_video_name: Original video filename
            frame_index: Index of frame with text (1-based)
            frame_extension: File extension for frame
            
        Returns:
            Generated filename
        """
        clean_name = self.clean_filename(Path(original_video_name).stem, max_length=30)
        return f"{clean_name}_{frame_index:04d}.{frame_extension}"
    
    def export_text_frames_from_analysis(self, 
                                       analysis_results: Dict[str, Any], 
                                       frames_source_dir: Optional[str] = None,
                                       copy_original_frames: bool = True) -> Dict[str, Any]:
        """
        Export frames with text based on analysis results
        
        Args:
            analysis_results: JSON analysis results from video pipeline
            frames_source_dir: Directory containing original frame files (if available)
            copy_original_frames: Whether to copy original frame files vs recreate them
            
        Returns:
            Export summary with details of exported frames
        """
        video_name = analysis_results.get('video_name', 'unknown_video')
        frames_with_text = analysis_results.get('frames_with_text', [])
        
        if not frames_with_text:
            return {
                'success': True,
                'exported_frames': 0,
                'export_folder': None,
                'message': 'No frames with text found'
            }
        
        # Create timestamped output folder
        export_folder = self.create_timestamped_folder(video_name)
        
        exported_frames = []
        successful_exports = 0
        
        for i, frame_data in enumerate(frames_with_text, 1):
            try:
                frame_path = frame_data.get('frame_path')
                timestamp = frame_data.get('timestamp', 0)
                extracted_text = frame_data.get('extracted_text', '')
                
                if not frame_path or not Path(frame_path).exists():
                    # Frame file not available
                    print(f"Warning: Frame file not found: {frame_path}")
                    continue
                
                # Generate new filename
                source_path = Path(frame_path)
                new_filename = self.generate_frame_filename(video_name, i, source_path.suffix.lstrip('.'))
                dest_path = export_folder / new_filename
                
                # Copy frame file
                shutil.copy2(source_path, dest_path)
                
                # Create metadata for this frame
                # Get OCR confidence if available
                ocr_data = frame_data.get('ocr', {})
                ocr_confidence = 0
                if ocr_data.get('region_results'):
                    confidences = [r.get('ocr_confidence', 0) for r in ocr_data['region_results']]
                    ocr_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                frame_info = {
                    'original_filename': source_path.name,
                    'exported_filename': new_filename,
                    'timestamp': timestamp,
                    'timestamp_str': frame_data.get('timestamp_str', ''),
                    'extracted_text': extracted_text,
                    'text_regions': frame_data.get('text_regions', 0),
                    'detection_confidence': frame_data.get('avg_confidence', 0),
                    'ocr_confidence': ocr_confidence
                }
                
                exported_frames.append(frame_info)
                successful_exports += 1
                
            except Exception as e:
                print(f"Error exporting frame {i}: {e}")
                continue
        
        # Create metadata file
        metadata = {
            'video_name': video_name,
            'export_timestamp': datetime.now().isoformat(),
            'total_frames_with_text': len(frames_with_text),
            'successfully_exported': successful_exports,
            'frames': exported_frames
        }
        
        metadata_file = export_folder / 'frame_export_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create simple text summary
        summary_file = export_folder / 'frame_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Exported Frames with Text: {video_name}\n")
            f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {successful_exports}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, frame_info in enumerate(exported_frames, 1):
                # Use the format: frame_number. MM:SS:    (detection confidence: X.XX, OCR confidence: X.XX)
                detection_confidence = frame_info.get('detection_confidence', 0)
                ocr_confidence = frame_info.get('ocr_confidence', 0)
                f.write(f"{i}. {frame_info['timestamp_str']}:      (detection confidence: {detection_confidence:.2f}, OCR confidence: {ocr_confidence:.2f})\n")
                if frame_info['extracted_text']:
                    f.write(f"{frame_info['extracted_text']}\n")
                else:
                    f.write("[No readable text extracted]\n")
                f.write("\n")
        
        return {
            'success': True,
            'exported_frames': successful_exports,
            'export_folder': str(export_folder),
            'metadata_file': str(metadata_file),
            'summary_file': str(summary_file),
            'frames': exported_frames
        }
    
    def export_frames_from_json(self, json_file: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Export frames from a JSON analysis file
        
        Args:
            json_file: Path to JSON analysis results
            output_dir: Output directory override
            
        Returns:
            Export summary
        """
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {json_path}")
        
        # Load analysis results
        with open(json_path, 'r') as f:
            analysis_results = json.load(f)
        
        # Override output directory if provided
        if output_dir:
            original_output_dir = self.base_output_dir
            self.base_output_dir = Path(output_dir)
        
        try:
            return self.export_text_frames_from_analysis(analysis_results)
        finally:
            # Restore original output directory
            if output_dir:
                self.base_output_dir = original_output_dir


def main():
    parser = argparse.ArgumentParser(description="Export video frames with detected text")
    parser.add_argument("--json", required=True, help="Path to analysis JSON file")
    parser.add_argument("--output", help="Output directory for exported frames")
    
    args = parser.parse_args()
    
    try:
        exporter = FrameExporter(args.output)
        result = exporter.export_frames_from_json(args.json, args.output)
        
        if result['success']:
            print(f"Successfully exported {result['exported_frames']} frames")
            print(f"Export folder: {result['export_folder']}")
            print(f"Metadata: {result['metadata_file']}")
            print(f"Summary: {result['summary_file']}")
        else:
            print("Export failed")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()