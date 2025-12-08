#!/usr/bin/env -S uv run
"""
Apple Vision Framework Text Detection Module

Uses macOS/iOS native Vision framework for fast text detection.
Optimal for Apple Silicon Macs with hardware acceleration.

Usage:
    uv run apple_vision_detector.py --image path/to/image.png
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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys

try:
    import Vision
    import Quartz
    from Foundation import NSURL
    from PIL import Image
except ImportError:
    print("Apple Vision Framework not available. Install dependencies:")
    print("uv add pyobjc-framework-Vision pyobjc-framework-Quartz pillow")
    sys.exit(1)


class AppleVisionTextDetector:
    """Text detection using Apple's Vision framework"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.request = None
        self._setup_vision_request()
    
    def _setup_vision_request(self):
        """Initialize Vision text detection request"""
        self.request = Vision.VNDetectTextRectanglesRequest.alloc().init()
        self.request.setReportCharacterBoxes_(True)  # Get character-level boxes
    
    def detect_text_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect text regions in an image using Apple Vision
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected text regions with bounding boxes and confidence
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image_url = NSURL.fileURLWithPath_(str(image_path))
        
        # Create image request handler
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, None)
        
        # Perform text detection
        success = handler.performRequests_error_([self.request], None)
        if not success[0]:
            raise RuntimeError("Vision text detection failed")
        
        # Process results
        observations = self.request.results()
        text_regions = []
        
        for observation in observations:
            confidence = float(observation.confidence())
            
            # Skip low-confidence detections
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box (Vision uses normalised coordinates)
            bbox = observation.boundingBox()
            
            # Convert to pixel coordinates (need image dimensions)
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Vision uses bottom-left origin, convert to top-left
            x = bbox.origin.x * width
            y = (1.0 - bbox.origin.y - bbox.size.height) * height
            w = bbox.size.width * width
            h = bbox.size.height * height
            
            region = {
                'bbox': [int(x), int(y), int(x + w), int(y + h)],  # [x1, y1, x2, y2]
                'bbox_normalised': [bbox.origin.x, bbox.origin.y, bbox.size.width, bbox.size.height],
                'confidence': confidence,
                'area': w * h,
                'detector': 'apple_vision'
            }
            
            # Add character boxes if available
            if hasattr(observation, 'characterBoxes') and observation.characterBoxes():
                char_boxes = []
                for char_box in observation.characterBoxes():
                    char_bbox = char_box.boundingBox()
                    char_x = char_bbox.origin.x * width
                    char_y = (1.0 - char_bbox.origin.y - char_bbox.size.height) * height
                    char_w = char_bbox.size.width * width
                    char_h = char_bbox.size.height * height
                    
                    char_boxes.append([
                        int(char_x), int(char_y), 
                        int(char_x + char_w), int(char_y + char_h)
                    ])
                
                region['character_boxes'] = char_boxes
            
            text_regions.append(region)
        
        return text_regions
    
    def has_text(self, image_path: str) -> bool:
        """Quick check if image contains any text"""
        regions = self.detect_text_regions(image_path)
        return len(regions) > 0
    
    def get_text_summary(self, image_path: str) -> Dict[str, Any]:
        """Get summary of text detection results"""
        regions = self.detect_text_regions(image_path)
        
        if not regions:
            return {
                'has_text': False,
                'text_regions': 0,
                'total_area': 0,
                'avg_confidence': 0,
                'detector': 'apple_vision'
            }
        
        total_area = sum(r['area'] for r in regions)
        avg_confidence = sum(r['confidence'] for r in regions) / len(regions)
        
        return {
            'has_text': True,
            'text_regions': len(regions),
            'total_area': total_area,
            'avg_confidence': avg_confidence,
            'regions': regions,
            'detector': 'apple_vision'
        }


def main():
    parser = argparse.ArgumentParser(description="Detect text using Apple Vision Framework")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--summary", action="store_true", help="Output summary only")
    parser.add_argument("--check-only", action="store_true", help="Only check if text exists")
    
    args = parser.parse_args()
    
    detector = AppleVisionTextDetector(args.confidence)
    
    try:
        if args.check_only:
            has_text = detector.has_text(args.image)
            print(json.dumps({"has_text": has_text}))
        elif args.summary:
            summary = detector.get_text_summary(args.image)
            print(json.dumps(summary, indent=2))
        else:
            regions = detector.detect_text_regions(args.image)
            result = {
                "image": args.image,
                "text_regions": regions,
                "detector": "apple_vision"
            }
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()