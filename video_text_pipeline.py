#!/usr/bin/env -S uv run
"""
Video Text Detection Pipeline

Orchestrates the complete pipeline: frame extraction -> text detection -> results aggregation.
Modular design allows swapping different text detection backends.

Usage:
    uv run video_text_pipeline.py --video path/to/video.mp4 --detector apple_vision --fps 1
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pillow",
#     "pyobjc-framework-Vision",
#     "pyobjc-framework-CoreML", 
#     "pyobjc-framework-Quartz",
# ]
# ///

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import sys
from datetime import datetime

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def generate_text_summary(analysis_data: Dict[str, Any], include_metadata: bool = True) -> str:
    """Generate human-readable text summary from analysis JSON"""
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

# Import our modules
sys.path.append(str(Path(__file__).parent / "modules"))
from frame_extractor import FrameExtractor
from apple_vision_detector import AppleVisionTextDetector
from apple_vision_ocr import AppleVisionOCR
from frame_exporter import FrameExporter

# Optional imports for additional detectors
try:
    from dbnet_detector import DBNetTinyDetector
    DBNET_AVAILABLE = True
except ImportError:
    DBNET_AVAILABLE = False


class VideoTextPipeline:
    """Main pipeline orchestrator for video text detection"""
    
    AVAILABLE_DETECTORS = {
        'apple_vision': AppleVisionTextDetector,
    }
    
    # Add optional detectors based on availability
    if DBNET_AVAILABLE:
        AVAILABLE_DETECTORS['dbnet_tiny'] = DBNetTinyDetector
    
    def __init__(self, 
                 detector_name: str = 'apple_vision', 
                 confidence_threshold: float = 0.5,
                 enable_ocr: bool = True,
                 ocr_level: str = 'accurate',
                 generate_summary: bool = False,
                 detailed_summary: bool = False,
                 export_text_frames: bool = False):
        self.detector_name = detector_name
        self.confidence_threshold = confidence_threshold
        self.enable_ocr = enable_ocr
        self.ocr_level = ocr_level
        self.generate_summary = generate_summary
        self.detailed_summary = detailed_summary
        self.export_text_frames = export_text_frames
        
        # Initialize components
        self.frame_extractor = FrameExtractor()
        self.text_detector = self._init_detector()
        self.ocr = AppleVisionOCR(recognition_level=ocr_level) if enable_ocr else None
        self.frame_exporter = FrameExporter() if export_text_frames else None
        
    def _init_detector(self):
        """Initialize the specified text detector"""
        if self.detector_name not in self.AVAILABLE_DETECTORS:
            raise ValueError(f"Unknown detector: {self.detector_name}. Available: {list(self.AVAILABLE_DETECTORS.keys())}")
        
        detector_class = self.AVAILABLE_DETECTORS[self.detector_name]
        return detector_class(confidence_threshold=self.confidence_threshold)
    
    def process_video(self, 
                     video_path: str, 
                     fps: float = 1.0, 
                     output_dir: Optional[str] = None,
                     cleanup_frames: bool = True) -> Dict[str, Any]:
        """
        Process entire video through the text detection pipeline
        
        Args:
            video_path: Path to input video
            fps: Frame extraction rate (frames per second)
            output_dir: Directory for intermediate files
            cleanup_frames: Whether to delete extracted frames after processing
            
        Returns:
            Complete analysis results
        """
        video_path = Path(video_path)
        start_time = time.time()
        
        # Setup output directory
        if output_dir:
            work_dir = Path(output_dir)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix=f"video_text_{video_path.stem}_"))
        
        work_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = work_dir / "frames"
        
        print(f"Processing video: {video_path.name}")
        print(f"Working directory: {work_dir}")
        
        try:
            # Step 1: Extract frames
            print(f"Extracting frames at {fps} FPS...")
            self.frame_extractor.output_dir = frames_dir
            frame_paths = self.frame_extractor.extract_frames(str(video_path), fps)
            
            if not frame_paths:
                raise RuntimeError("No frames extracted from video")
            
            # Step 2: Process each frame for text detection
            print(f"Analyzing {len(frame_paths)} frames...")
            frame_results = []
            frames_with_text = []
            
            for i, frame_path in enumerate(frame_paths):
                timestamp = i / fps  # Calculate timestamp based on frame index and FPS
                
                # Detect text in frame
                frame_result = self.text_detector.get_text_summary(frame_path)
                frame_result.update({
                    'frame_path': frame_path,
                    'frame_index': i,
                    'timestamp': timestamp,
                    'timestamp_str': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                })
                
                frame_results.append(frame_result)
                
                # Check if frame should be included based on text detection and OCR
                has_meaningful_text = False
                
                if frame_result['has_text']:
                    # Perform OCR if enabled and text was detected
                    if self.enable_ocr and self.ocr:
                        try:
                            ocr_result = self.ocr.extract_text_from_regions(
                                frame_path, 
                                frame_result.get('regions', [])
                            )
                            frame_result['ocr'] = ocr_result
                            frame_result['extracted_text'] = ocr_result.get('full_text', '')
                            
                            # Only consider it meaningful if OCR extracted actual text
                            if frame_result['extracted_text'].strip():
                                has_meaningful_text = True
                                
                        except Exception as e:
                            print(f"  OCR failed for frame {i}: {e}")
                            frame_result['ocr_error'] = str(e)
                            frame_result['extracted_text'] = ''
                    else:
                        # If OCR is disabled, rely on detection only
                        has_meaningful_text = True
                    
                    # Only add to frames_with_text if we have meaningful OCR results
                    if has_meaningful_text:
                        frames_with_text.append(frame_result)
                    
                # Progress indicator
                if (i + 1) % 10 == 0 or i == len(frame_paths) - 1:
                    print(f"  Processed {i + 1}/{len(frame_paths)} frames")
            
            # Step 3: Aggregate results
            processing_time = time.time() - start_time
            video_info = self.frame_extractor.get_video_info(video_path)
            
            results = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'processing_timestamp': datetime.now().isoformat(),
                'pipeline_config': {
                    'detector': self.detector_name,
                    'fps': fps,
                    'confidence_threshold': self.confidence_threshold
                },
                'video_info': {
                    'duration': float(video_info['format']['duration']),
                    'size_mb': round(int(video_info['format']['size']) / 1024 / 1024, 2),
                    'format': video_info['format']['format_name']
                },
                'analysis_summary': {
                    'total_frames': len(frame_results),
                    'frames_with_text': len(frames_with_text),
                    'text_coverage_percent': round(len(frames_with_text) / len(frame_results) * 100, 1),
                    'processing_time_seconds': round(processing_time, 2),
                    'frames_per_second': round(len(frame_results) / processing_time, 1)
                },
                'frames_with_text': frames_with_text,
                'all_frames': frame_results if len(frame_results) <= 100 else frame_results[:100]  # Limit output size
            }
            
            # Save detailed results
            results_file = work_dir / f"{video_path.stem}_text_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate text summary if requested
            if self.generate_summary or self.detailed_summary:
                summary_file = work_dir / f"{video_path.stem}_text_summary.txt"
                summary_text = generate_text_summary(results, include_metadata=True)
                
                with open(summary_file, 'w') as f:
                    f.write(summary_text)
                
                print(f"\\nAnalysis complete!")
                print(f"Found text in {len(frames_with_text)}/{len(frame_results)} frames ({results['analysis_summary']['text_coverage_percent']}%)")
                print(f"Processing time: {processing_time:.1f}s ({results['analysis_summary']['frames_per_second']:.1f} fps)")
                print(f"Results saved to: {results_file}")
                print(f"Text summary saved to: {summary_file}")
                
                # Also print the summary to console
                print(f"\\n{summary_text}")
            else:
                print(f"\\nAnalysis complete!")
                print(f"Found text in {len(frames_with_text)}/{len(frame_results)} frames ({results['analysis_summary']['text_coverage_percent']}%)")
                print(f"Processing time: {processing_time:.1f}s ({results['analysis_summary']['frames_per_second']:.1f} fps)")
                print(f"Results saved to: {results_file}")
            
            # Export frames with text if requested
            if self.export_text_frames and self.frame_exporter and frames_with_text:
                try:
                    self.frame_exporter.base_output_dir = work_dir.parent  # Export to parent of work_dir
                    export_result = self.frame_exporter.export_text_frames_from_analysis(results)
                    
                    if export_result['success']:
                        print(f"Exported {export_result['exported_frames']} frames with text to:")
                        print(f"  {export_result['export_folder']}")
                        print(f"  Metadata: {Path(export_result['metadata_file']).name}")
                        print(f"  Summary: {Path(export_result['summary_file']).name}")
                    else:
                        print("Frame export failed")
                        
                except Exception as e:
                    print(f"Frame export error: {e}")
            
            return results
            
        finally:
            # Cleanup extracted frames if requested
            if cleanup_frames and frames_dir.exists():
                import shutil
                shutil.rmtree(frames_dir)
                print(f"Cleaned up frame files")
    
    def quick_sample(self, video_path: str, num_samples: int = 5) -> Dict[str, Any]:
        """
        Quick sampling of video at regular intervals for initial assessment
        """
        video_info = self.frame_extractor.get_video_info(video_path)
        duration = float(video_info['format']['duration'])
        
        # Calculate sample timestamps
        timestamps = [duration * i / (num_samples - 1) for i in range(num_samples)]
        
        results = []
        for i, timestamp in enumerate(timestamps):
            try:
                # Extract single frame
                frame_path = self.frame_extractor.extract_frame_with_timestamp(
                    video_path, timestamp
                )
                
                # Verify frame exists
                if not Path(frame_path).exists():
                    print(f"Warning: Frame not created at {frame_path}")
                    continue
                
                # Analyze frame
                frame_result = self.text_detector.get_text_summary(frame_path)
                frame_result.update({
                    'sample_index': i,
                    'timestamp': timestamp,
                    'timestamp_str': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                })
                
                # Add OCR if text detected and OCR enabled
                if frame_result.get('has_text') and self.enable_ocr and self.ocr:
                    try:
                        ocr_result = self.ocr.extract_text_from_regions(
                            frame_path, 
                            frame_result.get('regions', [])
                        )
                        frame_result['ocr'] = ocr_result
                        frame_result['extracted_text'] = ocr_result.get('full_text', '')
                    except Exception as e:
                        print(f"OCR failed for sample at {timestamp}s: {e}")
                        frame_result['ocr_error'] = str(e)
                        frame_result['extracted_text'] = ''
                
                results.append(frame_result)
                
            except Exception as e:
                print(f"Error processing frame at {timestamp}s: {e}")
                continue
                
            finally:
                # Cleanup single frame if it exists
                if 'frame_path' in locals() and Path(frame_path).exists():
                    Path(frame_path).unlink()
        
        summary = {
            'video_path': video_path,
            'sample_type': 'quick_sample',
            'num_samples': num_samples,
            'samples_with_text': sum(1 for r in results if r['has_text']),
            'samples': results
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Video Text Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--detector", default="apple_vision", 
                       choices=VideoTextPipeline.AVAILABLE_DETECTORS.keys(),
                       help="Text detection backend to use")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output", help="Output directory for results and frames")
    parser.add_argument("--quick-sample", type=int, metavar="N", 
                       help="Quick sample N frames instead of full analysis")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frame files")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text extraction")
    parser.add_argument("--ocr-level", choices=['accurate', 'fast'], default='accurate',
                       help="OCR recognition level")
    parser.add_argument("--text-summary", action="store_true", 
                       help="Generate human-friendly text summary")
    parser.add_argument("--detailed-summary", action="store_true",
                       help="Generate detailed text summary with confidence scores")
    parser.add_argument("--export-frames", action="store_true",
                       help="Export frames with detected text to timestamped folder")
    parser.add_argument("--detection-threshold", type=float, default=0.5,
                       help="Minimum detection confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = VideoTextPipeline(
            detector_name=args.detector,
            confidence_threshold=args.confidence,
            enable_ocr=not args.no_ocr,
            ocr_level=args.ocr_level,
            generate_summary=args.text_summary,
            detailed_summary=args.detailed_summary,
            export_text_frames=args.export_frames
        )
        
        # Run analysis
        if args.quick_sample:
            results = pipeline.quick_sample(str(video_path), args.quick_sample)
            print(json.dumps(results, indent=2))
        else:
            results = pipeline.process_video(
                str(video_path), 
                fps=args.fps,
                output_dir=args.output,
                cleanup_frames=not args.keep_frames
            )
            
            # Print summary
            summary = results['analysis_summary']
            print(f"\\n=== SUMMARY ===")
            print(f"Video: {results['video_name']}")
            print(f"Duration: {results['video_info']['duration']:.1f}s")
            print(f"Frames analyzed: {summary['total_frames']}")
            print(f"Frames with text: {summary['frames_with_text']} ({summary['text_coverage_percent']}%)")
            print(f"Detection rate: {summary['frames_per_second']:.1f} fps")
        
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()