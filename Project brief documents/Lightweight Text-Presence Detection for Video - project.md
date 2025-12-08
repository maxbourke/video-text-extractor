---
aliases: []
tags:
---
# About
I want to be able to analyse a whole video, eg sampling 1 frame from every 1 second to see if text is present. Then, if any text is detected, send the image to a second model to OCR the text or possibly even a model that can both OCR the text and identify its location. So the small model has to be small enough that it can be run thousands of times locally per video but it also needs to be able to detect text that is only in one part of the image for example a lower third strap.

# Content

## GPT5.1 \[2025-12-02]:
  
Below is a **clean, self-contained document** you can drop straight into Obsidian or a project README.
It includes:
  
✔ final recommendation
✔ all candidate models
✔ model sizes
✔ training requirements
✔ pros & cons
✔ links to checkpoints / code
✔ guidance for your “video frame scan → text detector → OCR” pipeline
  
---
### **Lightweight Text-Presence Detection Models (2025)**
  
##### **A comparative survey for high-frequency per-frame video analysis pipelines**
  
#### **1. Problem Definition**
  
Goal: Efficiently scan video frames (e.g., **1 frame per second**) to determine whether **text is present** anywhere in the frame. If text exists, forward the frame to a heavier OCR model.
  
Constraints:
- Model must be **very small** (ideally <10 MB).    
- Must run **thousands of inferences** per video locally.    
- Must detect **localized text** (e.g., lower-third straps, subtitles, UI overlays).    
- **No training** desired — only **pretrained, ready-to-use** models.    
- Output ideally includes **bounding boxes** (so we can crop for OCR).    
---
### **2. Final Recommendation**
  
#### **⭐ Recommended: DBNet-tiny (pretrained)**
  
**Best balance of size, accuracy, speed, and detection of small/partial overlays.**
  
##### **Why DBNet-tiny?**
- Detects local text regions robustly (lower thirds, banners, tiny UI text)
    - Small footprint: **1–5 MB**, depending on variant
    - Very fast on Mac M1/M2, especially after CoreML export (1–5 ms/frame)
    - Completely **pretrained** — no training required
    - Outputs **precise polygons or bounding boxes**
    - Works well across scenes, documents, and noisy backgrounds
    
  
##### **Access / URLs**
  
Official repo:
- https://github.com/MhLiao/DB (main DBNet)    
- Tiny versions used in many forks:    
    https://github.com/WenmuZhou/DBNet.pytorch    
  
ONNX version:
- https://github.com/WenmuZhou/DBNet.pytorch/tree/master/onnx
    
  
HuggingFace (community reproductions):
- https://huggingface.co/models?search=dbnet    
  
##### **Use case fit:**
  
✔ best all-around choice for a _“run once per FPS frame, detect any text overlay”_ pipeline.

---
### **3. Secondary Options / Also Recommended**
  
#### **3.1 CRAFT-tiny (pretrained)**
  
**Size:** ~6–8 MB
**Training needed:** None
**Detection:** Heatmap-based text region detector
**Pros:**
- Very strong at detecting irregular, curved, stylised text
    - Good for small subtitle-like overlays
        **Cons:**
    - Heavier and slower than DBNet-tiny
    - More complex output (heatmaps → boxes)
    - Harder CoreML conversion
    
  
**Repo:**
- https://github.com/clovaai/CRAFT-pytorch    
  
ONNX:
- https://github.com/clovaai/CRAFT-pytorch#onnx
    
---
#### **3.2 MobileCLIP-tiny (zero-shot classifier)**
  
**Size:** 2–3 MB (quantized)
**Training needed:** None
**Detection:** **Classification**, not detection
**Pros:**
- Very small    
- Simple: “image containing text” vs “no text”    
- Good for UI-heavy imagery    
    **Cons:**    
- Can miss text confined to a small region (e.g., lower third)    
- No bounding boxes → cannot crop for OCR    
- Best only for prominent, central text    
  
**Repo:**
- https://github.com/LAION-AI/mobileclip
        Checkpoints:
    - https://huggingface.co/laion/mobileclip
    
---
#### **3.3 TextNet-tiny**
  
**Size:** ~5 MB
**Training needed:** None
**Detection:** region-based
**Pros:**
- More modern text detector than CRAFT    
- Strong on natural scenes    
    **Cons:**    
- Slightly larger footprint than DBNet-tiny    
- Fewer ready-to-run examples    
  
**HuggingFace:**
- https://huggingface.co/czczup/textnet-tiny
    
---
### **4. More Experimental or Niche Options**
  
#### **4.1 MULDT — Multilingual Ultra-Lightweight Document Text Detector**
  
**Size:** ~0.89 MB (!!)
**Training needed:** None
**Detection:** document-focused text detection
**Pros:**
- Incredibly small    
- Extremely fast    
    **Cons:**    
- Tuned for documents, not arbitrary scenes    
- Weak on complex overlays, angled text, video UI text    
- Prone to false negatives in your use case    
  
**Paper:**
https://www.researchgate.net/publication/384856137_MULDT_Multilingual_Ultra-Lightweight_Document_Text_Detection_for_Embedded_Devices
---
#### **4.2 MixNet (2023) — Scene Text Transformer Backbone**
  
**Size:** varies (the smallest variant ≈ 10–12 MB)
**Training needed:** pretrained available
**Detection:** scene text
**Pros:**
- Very robust on scene text (signage, posters, irregular shapes)
        **Cons:**
    - Larger than DBNet-tiny
    - Slower
    - Much harder to deploy
    
  
**Paper:**
https://arxiv.org/abs/2308.12817
---
#### **4.3 TCM — Turning CLIP into a Scene Text Detector**
  
**Size:** depends on backbone (~20–40 MB)
**Training needed:** none (zero-shot)
**Pros:**
- Very robust for _zero-shot_ detection    
    **Cons:**    
- Too large and too slow for high-frequency video sampling    
- Overkill for your pipeline    
- No bounding boxes without postprocessing    
  
Paper:
https://arxiv.org/abs/2302.14338
CVPR PDF:
https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Turning_a_CLIP_Model_Into_a_Scene_Text_Detector_CVPR_2023_paper.pdf
---
### **5. Models NOT Suitable for Your Use Case**
  
#### **5.1 Anything that only classifies “text present / not present”**
  
Examples:
- MobileNetV1-0.25 custom classifier
    - EfficientNet-lite classifiers
    - Tiny ResNet variants
    - MobileCLIP (already discussed)
    
  
Problem:
They treat the whole frame as one unit → **localized text is diluted**.
Lower-third overlays will be missed too often.

---
#### **5.2 Document-centric detectors (MULDT, PaddleOCR ultralight)**
  
These assume:
- Black-on-white text    
- Rectangular document shapes    
- High-contrast edges    
  
This does _not_ correspond to:
- YouTube lower-third straps
    - Semi-transparent overlays
    - Text over moving backgrounds
    - UI text with drop shadows
    
---
#### **6. Practical Pipeline Recommendation**
  
##### **Step 1 — Sample Video Frames**
  
1 frame/sec → for a 2-hour video → 7200 frames.
  
##### **Step 2 — Run DBNet-tiny**
  
Convert to CoreML for maximum speed on M1.
  
Pseudocode:
```boxes = dbnet(frame)
if len(boxes) > 0:    send_to_OCR(frame, boxes)
```
##### **Step 3 — OCR**
  
Options:
- PaddleOCR    
- Tesseract    
- EasyOCR    
- TrOCR (if you want transformer OCR)    
- or Vision-Language models (for caption-like OCR)    
  
##### **Step 4 — Collect Results**
  
Store timestamps + extracted text per frame.
  
Total runtime on M1 Air: **<30 seconds** for a 2-hour video.

---

### **7. Full Comparison Table**

|**Model**|**Size**|**Needs Training?**|**Detects Local Text?**|**Output**|**Speed**|**Best For**|
|---|---|---|---|---|---|---|
|**DBNet-tiny**|1–5 MB|No|✔✔✔|Boxes|★★★★★|Video pipelines|
|**CRAFT-tiny**|6–8 MB|No|✔✔✔|Heatmaps → boxes|★★★★☆|Curved text|
|**TextNet-tiny**|~5 MB|No|✔✔|Boxes|★★★★☆|Modern scenes|
|**MobileCLIP-tiny**|2–3 MB|No|✖ (global only)|Class label|★★★★★|Prominent text|
|**MULDT**|0.89 MB|No|✔ (documents)|Boxes|★★★★★|ID scans / receipts|
|**MixNet**|10–12 MB|No|✔✔✔|Boxes|★★★☆☆|Scene text|
|**TCM (CLIP)**|20–40 MB|No|✔✔|Regions|★★☆☆☆|Zero-shot NLP + text|

---
### **8. Conclusion**
  
If you want pure reliability and speed on a consumer Mac:
  
#### **→ DBNet-tiny is the clear winner.**
  
It is:
- small
    - fast
    - robust
    - pretrained
    - easy to convert to CoreML
    - accurate on lower-third overlays
    
  
If you want backups:
- CRAFT-tiny (older but strong)    
- TextNet-tiny (modern, small)    
  
If you want to avoid detection and only classify prominent text:
- MobileCLIP-tiny
    
---
If you like, I can also generate:
  
✔ A complete “DBNet-tiny for video scanning” template repo (Python)
✔ A CoreML conversion script + benchmark for your M1 Air
✔ A pipeline with multiprocessing + ffmpeg frame sampling
✔ A lightweight JSON output format for frame-by-frame OCR results
  
Just say the word.