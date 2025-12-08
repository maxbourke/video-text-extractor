#!/usr/bin/env -S uv run
"""
Apple Vision Framework OCR Module

Uses macOS/iOS native Vision framework for text recognition and extraction.
Designed to work with frames that have already been identified as containing text.

Usage:
    uv run apple_vision_ocr.py --image path/to/image.png
    uv run apple_vision_ocr.py --image path/to/image.png --crop 100,100,200,200
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


class AppleVisionOCR:
    """OCR text extraction using Apple's Vision framework"""
    
    def __init__(self, 
                 recognition_level: str = 'accurate',
                 language_correction: bool = True,
                 languages: Optional[List[str]] = None,
                 bbox_expansion: int = 5):
        """
        Initialize OCR with configuration
        
        Args:
            recognition_level: 'accurate' or 'fast' 
            language_correction: Enable language correction
            languages: List of language codes (e.g., ['en-US', 'en-GB'])
            bbox_expansion: Pixels to expand bounding boxes by (helps capture edge characters)
        """
        self.recognition_level = recognition_level
        self.language_correction = language_correction
        self.languages = languages or ['en-US']
        self.bbox_expansion = bbox_expansion
        
    def _create_text_request(self):
        """Create and configure VNRecognizeTextRequest"""
        request = Vision.VNRecognizeTextRequest.alloc().init()
        
        # Set recognition level
        if self.recognition_level == 'accurate':
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        else:
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        
        # Enable language correction
        request.setUsesLanguageCorrection_(self.language_correction)
        
        # Set supported languages
        if self.languages:
            request.setRecognitionLanguages_(self.languages)
        
        return request
    
    def _expand_bbox(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Expand bounding box by self.bbox_expansion pixels in all directions
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            img_width: Image width for boundary checking
            img_height: Image height for boundary checking
            
        Returns:
            Expanded bounding box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        expansion = self.bbox_expansion
        
        # Expand and clamp to image boundaries
        expanded_x1 = max(0, x1 - expansion)
        expanded_y1 = max(0, y1 - expansion)
        expanded_x2 = min(img_width, x2 + expansion)
        expanded_y2 = min(img_height, y2 + expansion)
        
        return (expanded_x1, expanded_y1, expanded_x2, expanded_y2)
    
    def extract_text_from_image(self, image_path: str, crop_box: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Extract all text from an image
        
        Args:
            image_path: Path to image file
            crop_box: Optional crop box (x1, y1, x2, y2) to extract text from specific region
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Crop image if requested
        if crop_box:
            with Image.open(image_path) as img:
                cropped_img = img.crop(crop_box)
                # Save cropped image temporarily
                temp_path = image_path.parent / f"temp_cropped_{image_path.name}"
                cropped_img.save(temp_path)
                image_url = NSURL.fileURLWithPath_(str(temp_path))
        else:
            image_url = NSURL.fileURLWithPath_(str(image_path))
        
        # Create request handler
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, None)
        
        # Create text recognition request
        request = self._create_text_request()
        
        try:
            # Perform OCR
            success = handler.performRequests_error_([request], None)
            if not success[0]:
                raise RuntimeError("Vision text recognition failed")
            
            # Process results
            observations = request.results()
            extracted_text = []
            full_text_lines = []
            
            for observation in observations:
                # Get top candidate text
                top_candidates = observation.topCandidates_(1)
                if not top_candidates or len(top_candidates) == 0:
                    continue
                
                candidate = top_candidates[0]
                text = str(candidate.string())
                confidence = float(candidate.confidence())
                
                # Get bounding box
                bbox = observation.boundingBox()
                
                # Convert to pixel coordinates if we have original image
                if crop_box:
                    # For cropped images, adjust coordinates
                    img_width = crop_box[2] - crop_box[0]
                    img_height = crop_box[3] - crop_box[1]
                    x = bbox.origin.x * img_width + crop_box[0]
                    y = (1.0 - bbox.origin.y - bbox.size.height) * img_height + crop_box[1]
                    w = bbox.size.width * img_width
                    h = bbox.size.height * img_height
                else:
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                    x = bbox.origin.x * img_width
                    y = (1.0 - bbox.origin.y - bbox.size.height) * img_height
                    w = bbox.size.width * img_width
                    h = bbox.size.height * img_height
                
                text_item = {
                    'text': text,
                    'confidence': confidence,
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'bbox_normalised': [bbox.origin.x, bbox.origin.y, bbox.size.width, bbox.size.height]
                }
                
                extracted_text.append(text_item)
                full_text_lines.append(text)
            
            # Clean up temp file if created
            if crop_box and temp_path.exists():
                temp_path.unlink()
            
            # Combine all text
            full_text = '\n'.join(full_text_lines)
            
            return {
                'image_path': str(image_path),
                'crop_box': crop_box,
                'total_text_blocks': len(extracted_text),
                'full_text': full_text,
                'text_blocks': extracted_text,
                'avg_confidence': sum(item['confidence'] for item in extracted_text) / len(extracted_text) if extracted_text else 0,
                'ocr_engine': 'apple_vision'
            }
            
        except Exception as e:
            # Clean up temp file on error
            if crop_box and 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    def extract_text_from_regions(self, image_path: str, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract text from specific regions detected by text detection
        
        Args:
            image_path: Path to image file
            text_regions: List of text regions with bounding boxes from text detection
            
        Returns:
            Dictionary containing text extracted from each region
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not text_regions:
            return {
                'image_path': str(image_path),
                'total_regions': 0,
                'regions_with_text': 0,
                'full_text': '',
                'region_results': [],
                'ocr_engine': 'apple_vision'
            }
        
        region_results = []
        all_text_lines = []
        
        # Get image dimensions for bbox expansion
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        for i, region in enumerate(text_regions):
            bbox = region.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            
            try:
                # Expand bounding box to capture edge characters
                expanded_bbox = self._expand_bbox(tuple(bbox), img_width, img_height)
                
                # Extract text from the expanded region
                result = self.extract_text_from_image(image_path, expanded_bbox)
                
                region_result = {
                    'region_index': i,
                    'detection_bbox': bbox,
                    'expanded_bbox': list(expanded_bbox),
                    'detection_confidence': region.get('confidence', 0),
                    'extracted_text': result.get('full_text', ''),
                    'text_blocks': result.get('text_blocks', []),
                    'ocr_confidence': result.get('avg_confidence', 0)
                }
                
                region_results.append(region_result)
                
                if result.get('full_text'):
                    all_text_lines.append(result['full_text'])
                    
            except Exception as e:
                print(f"Failed to extract text from region {i}: {e}")
                region_results.append({
                    'region_index': i,
                    'detection_bbox': bbox,
                    'extraction_error': str(e)
                })
        
        full_text = '\n'.join(all_text_lines)
        regions_with_text = len([r for r in region_results if r.get('extracted_text')])
        
        return {
            'image_path': str(image_path),
            'total_regions': len(text_regions),
            'regions_with_text': regions_with_text,
            'full_text': full_text,
            'region_results': region_results,
            'ocr_engine': 'apple_vision'
        }
    
    def quick_extract(self, image_path: str) -> str:
        """Quick text extraction returning just the combined text"""
        result = self.extract_text_from_image(image_path)
        return result.get('full_text', '')


def main():
    parser = argparse.ArgumentParser(description="Extract text using Apple Vision OCR")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--crop", help="Crop box as x1,y1,x2,y2")
    parser.add_argument("--level", choices=['accurate', 'fast'], default='accurate', 
                       help="Recognition level")
    parser.add_argument("--languages", nargs='+', default=['en-US'], 
                       help="Language codes (e.g., en-US fr-FR)")
    parser.add_argument("--no-correction", action="store_true", 
                       help="Disable language correction")
    parser.add_argument("--bbox-expansion", type=int, default=5,
                       help="Pixels to expand bounding boxes by (default: 5)")
    parser.add_argument("--quick", action="store_true", help="Quick extraction (text only)")
    
    args = parser.parse_args()
    
    # Parse crop box
    crop_box = None
    if args.crop:
        try:
            coords = [int(x.strip()) for x in args.crop.split(',')]
            if len(coords) == 4:
                crop_box = tuple(coords)
            else:
                raise ValueError("Crop box must have 4 coordinates")
        except ValueError as e:
            print(f"Invalid crop box: {e}", file=sys.stderr)
            sys.exit(1)
    
    try:
        ocr = AppleVisionOCR(
            recognition_level=args.level,
            language_correction=not args.no_correction,
            languages=args.languages,
            bbox_expansion=args.bbox_expansion
        )
        
        if args.quick:
            text = ocr.quick_extract(args.image)
            print(text)
        else:
            result = ocr.extract_text_from_image(args.image, crop_box)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"OCR Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()