"""
suspect_finder.py
=================
Suspect Finder Module — Part of Smart Surveillance Framework
------------------------------------------------------------
Just edit the CONFIG block below and run:  python suspect_finder.py

Dependencies:  pip install ultralytics opencv-python numpy requests
Groq free API key (for better description parsing): https://console.groq.com
"""

# ══════════════════════════════════════════════════════════════
#  ✏️  CONFIG — edit these variables and run the script
# ══════════════════════════════════════════════════════════════

# Input: set VIDEO_PATH to your video file, or IMAGE_PATH to a single image.
# Leave the one you're NOT using as an empty string "".
VIDEO_PATH  = "./videos/cam4.1.mp4"          # e.g. r"C:\footage\cctv.mp4"
IMAGE_PATH  = ""                   # e.g. r"C:\footage\frame.jpg"

# Describe the suspect in plain English
DESCRIPTION = "green scarf, black jacket, brown bag"

# Groq API key for smarter description parsing (free at https://console.groq.com)
# Leave as "" to use the built-in rule-based parser instead
GROQ_API_KEY = "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"

# YOLO model: "yolov8n.pt" (fastest) → "yolov8s.pt" → "yolov8m.pt" → "yolov8l.pt" (most accurate)
YOLO_MODEL      = "yolov8n.pt"
YOLO_CONFIDENCE = 0.40             # Detection confidence threshold (0.0 – 1.0)

# How many frames to skip between detections (5 = process every 5th frame)
# Lower = more thorough but slower.  Higher = faster but may miss people.
SKIP_FRAMES = 5

# Match sensitivity: lower = more results, higher = stricter
# 0.15 = loose (many results)   0.35 = strict (only strong matches)
MATCH_THRESHOLD = 0.20

# Maximum number of matches to show
TOP_N = 20

# Output folder for results grid image + JSON report
OUTPUT_DIR  = "results"
SAVE_CROPS  = False    # True = also save each match as individual crop image
NO_DISPLAY  = False    # True = skip OpenCV window (just save to file)

# ══════════════════════════════════════════════════════════════
#  END CONFIG — no need to edit below this line
# ══════════════════════════════════════════════════════════════

import cv2
import numpy as np
import json
import re
import os
import sys
import time
import requests
from dataclasses import dataclass, field
from datetime import timedelta
from collections import Counter
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — COLOR ANALYZER
#  HSV-based color detection with shade-tolerant fuzzy matching
# ══════════════════════════════════════════════════════════════

# HSV color definitions — (hue_center, hue_tolerance, min_sat, min_val, max_val)
# Hue range in OpenCV: 0-179
_COLOR_RANGES = {
    "red":    [(0, 10, 60, 40, 255), (165, 10, 60, 40, 255)],  # wraps 0/180
    "orange": [(10, 10, 80, 80, 255)],
    "yellow": [(22, 10, 80, 80, 255)],
    "green":  [(60, 25, 50, 40, 255)],
    "cyan":   [(90, 10, 50, 40, 255)],
    "blue":   [(110, 20, 50, 30, 255)],   # catches navy, royal, sky
    "purple": [(135, 15, 40, 30, 255)],
    "pink":   [(160, 10, 40, 100, 255)],
    "white":  [(0, 180, 0, 180, 255)],    # handled separately
    "black":  [(0, 180, 0, 0, 80)],       # handled separately
    "gray":   [(0, 180, 0, 80, 180)],     # handled separately
    "brown":  [(12, 8, 60, 30, 180)],
    "beige":  [(18, 8, 20, 150, 230)],
    "khaki":  [(20, 8, 30, 120, 200)],
    "navy":   [(110, 15, 50, 20, 100)],
    "maroon": [(0, 8, 60, 30, 100)],
}

_COLOR_ALIASES = {
    "dark blue": "navy", "light blue": "blue", "sky blue": "blue",
    "royal blue": "blue", "cobalt": "blue", "indigo": "purple",
    "violet": "purple", "magenta": "pink", "scarlet": "red",
    "crimson": "red", "burgundy": "maroon", "olive": "green",
    "dark green": "green", "lime": "green", "teal": "cyan",
    "turquoise": "cyan", "tan": "beige", "cream": "beige",
    "off-white": "white", "charcoal": "gray", "dark gray": "gray",
    "silver": "gray", "grey": "gray",
}


def _normalize_color(name: str) -> str:
    name = name.lower().strip()
    return _COLOR_ALIASES.get(name, name)


def _create_color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    color = _normalize_color(color)
    if color == "white":
        return cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    if color == "black":
        return cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    if color == "gray":
        return cv2.inRange(hsv, (0, 0, 80), (180, 40, 180))

    ranges = _COLOR_RANGES.get(color, [])
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (hc, ht, ms, mv, xv) in ranges:
        lo = np.array([max(0, hc - ht), ms, mv])
        hi = np.array([min(179, hc + ht), 255, xv])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def _get_dominant_colors(region: np.ndarray, top_n: int = 3) -> list:
    """Returns [(color_name, coverage_pct), ...] for region."""
    if region is None or region.size == 0:
        return []
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return []
    scores = {c: np.sum(_create_color_mask(hsv, c) > 0) / total
              for c in _COLOR_RANGES}
    return [(c, round(v * 100, 1))
            for c, v in sorted(scores.items(), key=lambda x: -x[1])
            if v > 0.05][:top_n]


def _color_match_score(region: np.ndarray, target_color: str) -> float:
    """Score 0.0–1.0: how well region matches target_color."""
    if region is None or region.size == 0:
        return 0.0
    h, w = region.shape[:2]
    if h < 20 or w < 20:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    mask = _create_color_mask(hsv, _normalize_color(target_color))
    return min(1.0, np.sum(mask > 0) / total * 1.5)


def _extract_zones(crop: np.ndarray) -> dict:
    """Split person crop into body zones."""
    h, w = crop.shape[:2]
    if h < 40 or w < 10:
        return {}
    return {
        "head":       crop[0:int(h * 0.15), :],
        "upper_body": crop[int(h * 0.15):int(h * 0.55), :],
        "lower_body": crop[int(h * 0.55):int(h * 0.90), :],
        "shoes":      crop[int(h * 0.90):, :],
    }


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — DESCRIPTION PARSER
#  Groq LLaMA API + rule-based fallback
# ══════════════════════════════════════════════════════════════

_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"

_PARSER_SYSTEM = """You are a forensic description parser. Extract clothing attributes from a suspect description into strict JSON.

Output ONLY valid JSON, no explanation, no markdown:
{
  "upper_body":  {"color": "blue",  "type": "jacket"} or null,
  "lower_body":  {"color": "black", "type": "pants"}  or null,
  "shoes":       {"color": "white", "type": "sneakers"} or null,
  "hat":         {"color": "red",   "type": "cap"} or null,
  "accessories": [{"color": "brown", "type": "bag"}]   or [],
  "gender": "male" or "female" or null,
  "build":  "slim" or "heavy" or "medium" or null
}
Normalize colors to: red orange yellow green blue purple pink white black gray brown beige khaki navy maroon"""


def _parse_groq(description: str, api_key: str) -> dict:
    resp = requests.post(
        _GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": _GROQ_MODEL,
            "messages": [
                {"role": "system", "content": _PARSER_SYSTEM},
                {"role": "user",   "content": f'Parse: "{description}"'}
            ],
            "temperature": 0.1,
            "max_tokens": 400
        },
        timeout=15
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def _parse_fallback(description: str) -> dict:
    """Rule-based parser — no API needed."""
    desc = description.lower()

    COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
              "white", "black", "gray", "grey", "brown", "navy", "maroon",
              "beige", "khaki", "tan", "cyan", "teal"]
    UPPER  = ["jacket", "coat", "hoodie", "sweatshirt", "shirt", "t-shirt",
              "tshirt", "sweater", "blazer", "top", "blouse", "vest"]
    LOWER  = ["pants", "jeans", "trousers", "shorts", "skirt", "leggings",
              "joggers", "chinos", "slacks"]
    BAGS   = ["bag", "backpack", "purse", "handbag", "suitcase",
              "briefcase", "tote", "satchel", "duffel"]
    SHOES  = ["shoes", "sneakers", "boots", "heels", "sandals",
              "loafers", "trainers"]
    HATS   = ["hat", "cap", "beanie", "hood", "helmet", "beret"]

    def find_color(keyword):
        pat = rf'(\b(?:{"|".join(COLORS)})\b)\s+(?:\w+\s+)?{keyword}'
        m = re.search(pat, desc)
        return m.group(1) if m else "unknown"

    result = {"upper_body": None, "lower_body": None, "shoes": None,
              "hat": None, "accessories": [], "gender": None, "build": None}

    for t in UPPER:
        if t in desc:
            result["upper_body"] = {"color": find_color(t), "type": t}; break
    for t in LOWER:
        if t in desc:
            result["lower_body"] = {"color": find_color(t), "type": t}; break
    for t in BAGS:
        if t in desc:
            result["accessories"].append({"color": find_color(t), "type": t})
    for t in SHOES:
        if t in desc:
            result["shoes"] = {"color": find_color(t), "type": t}; break
    for t in HATS:
        if t in desc:
            result["hat"] = {"color": find_color(t), "type": t}; break

    if any(w in desc for w in ["man", "male", "guy", "boy"]):
        result["gender"] = "male"
    elif any(w in desc for w in ["woman", "female", "girl", "lady"]):
        result["gender"] = "female"

    return result


def parse_description(description: str, api_key: str = "") -> dict:
    """
    Parse a natural language suspect description into structured attributes.
    Uses Groq LLaMA API if key provided, else rule-based fallback.
    """
    if api_key and api_key.strip():
        print("[Parser] Using Groq LLaMA API...")
        try:
            result = _parse_groq(description, api_key)
            print(f"[Parser] {json.dumps(result)}")
            return result
        except Exception as e:
            print(f"[Parser] API failed ({e}), falling back to rule-based parser.")
    print("[Parser] Using rule-based fallback parser...")
    result = _parse_fallback(description)
    print(f"[Parser] {json.dumps(result)}")
    return result


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — YOLO DETECTOR
#  Person + bag detection with zone association
# ══════════════════════════════════════════════════════════════

_PERSON_CLS = 0
_BAG_CLS = {24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase"}


@dataclass
class Detection:
    person_id:       int
    frame_num:       int
    bbox:            tuple        # (x1, y1, x2, y2)
    confidence:      float
    crop:            np.ndarray = field(repr=False)
    carried_objects: list       = field(default_factory=list)
    timestamp:       float      = 0.0


class YOLODetector:
    def __init__(self, model_size: str = "yolov8m.pt", confidence: float = 0.4,
                 device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")
        print(f"[Detector] Loading {model_size}...")
        self.model = YOLO(model_size)
        self.confidence = confidence
        self.device = device

    def _near(self, pb: tuple, ob: tuple, ratio: float = 0.3) -> bool:
        px1, py1, px2, py2 = pb
        ox1, oy1, ox2, oy2 = ob
        pw, ph = px2 - px1, py2 - py1
        return (ox1 < px2 + pw * ratio and ox2 > px1 - pw * ratio and
                oy1 < py2 + ph * ratio and oy2 > py1 - ph * ratio)

    def detect_frame(self, frame: np.ndarray, frame_num: int = 0,
                     timestamp: float = 0.0) -> list:
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[_PERSON_CLS] + list(_BAG_CLS.keys()),
            verbose=False,
            device=self.device
        )[0]

        if results.boxes is None:
            return []

        H, W = frame.shape[:2]
        persons, bags = [], []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls == _PERSON_CLS:
                persons.append({"bbox": (x1, y1, x2, y2), "conf": conf})
            elif cls in _BAG_CLS:
                bag_crop = frame[max(0,y1):y2, max(0,x1):x2].copy()
                bags.append({"bbox": (x1, y1, x2, y2),
                             "type": _BAG_CLS[cls],
                             "conf": conf,
                             "crop": bag_crop})

        detections = []
        for pid, p in enumerate(persons):
            x1, y1, x2, y2 = p["bbox"]
            pad = 10
            crop = frame[max(0,y1-pad):min(H,y2+pad),
                         max(0,x1-pad):min(W,x2+pad)].copy()
            if crop.size == 0:
                continue
            nearby_bags = [b for b in bags if self._near(p["bbox"], b["bbox"])]
            detections.append(Detection(
                person_id=pid, frame_num=frame_num,
                bbox=p["bbox"], confidence=p["conf"],
                crop=crop, carried_objects=nearby_bags, timestamp=timestamp
            ))
        return detections

    def detect_video(self, video_path: str, skip_frames: int = 5,
                     max_frames: int = None, progress_cb=None) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"[Detector] {total} frames @ {fps:.1f} FPS — processing every {skip_frames}")

        all_dets, frame_num, processed = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % skip_frames == 0:
                dets = self.detect_frame(frame, frame_num, frame_num / fps)
                all_dets.extend(dets)
                processed += 1
                if progress_cb:
                    progress_cb(frame_num, total)
                if max_frames and processed >= max_frames:
                    break
            frame_num += 1

        cap.release()
        print(f"[Detector] Done — {len(all_dets)} detections in {processed} frames")
        return all_dets


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — MATCHER
#  Scores detections against parsed suspect attributes
# ══════════════════════════════════════════════════════════════

_ZONE_WEIGHTS = {
    "upper_body":  0.35,
    "lower_body":  0.30,
    "accessories": 0.25,
    "shoes":       0.05,
    "hat":         0.05,
}

MATCH_THRESHOLD = 0.20


@dataclass
class MatchResult:
    detection_index: int
    frame_num:       int
    timestamp:       float
    bbox:            tuple
    crop:            np.ndarray = field(repr=False)
    overall_score:   float      = 0.0
    zone_scores:     dict       = field(default_factory=dict)
    dominant_colors: dict       = field(default_factory=dict)
    carried_objects: list       = field(default_factory=list)


def _score_zone(zone_img, target_color: str) -> float:
    if zone_img is None or zone_img.size == 0:
        return 0.0
    if not target_color or target_color in ("unknown", "none", "null"):
        return 0.5  # neutral — no expectation
    return _color_match_score(zone_img, target_color)


def _score_accessories(detection: Detection, targets: list) -> tuple:
    if not targets:
        return 0.5, []
    if not detection.carried_objects:
        return 0.0, []

    best, matched = 0.0, []
    bag_words = {"bag", "backpack", "purse", "pack", "suitcase", "tote"}

    for tgt in targets:
        tgt_color = tgt.get("color", "unknown")
        tgt_type  = tgt.get("type", "bag").lower()

        for obj in detection.carried_objects:
            type_ok = (tgt_type in obj["type"] or obj["type"] in tgt_type or
                       any(w in tgt_type for w in bag_words))
            obj_crop = obj.get("crop")
            color_s  = _score_zone(obj_crop, tgt_color) if obj_crop is not None else 0.3
            score    = 0.3 * float(type_ok) + 0.7 * color_s
            if score > best:
                best = score
                matched = [{"target": tgt, "detected": obj, "score": score}]

    return best, matched


def _match_one(detection: Detection, attributes: dict) -> MatchResult:
    zones   = _extract_zones(detection.crop)
    z_scores, dom_colors = {}, {}
    weighted, weight_used = 0.0, 0.0

    def apply(zone_key, attr_key, weight):
        nonlocal weighted, weight_used
        attr = attributes.get(attr_key)
        if not attr:
            return
        zone = zones.get(zone_key)
        color = attr.get("color", "unknown") if isinstance(attr, dict) else "unknown"
        s = _score_zone(zone, color)
        z_scores[attr_key] = s
        dom_colors[attr_key] = _get_dominant_colors(zone) if zone is not None else []
        weighted    += s * weight
        weight_used += weight

    apply("upper_body", "upper_body", _ZONE_WEIGHTS["upper_body"])
    apply("lower_body", "lower_body", _ZONE_WEIGHTS["lower_body"])
    apply("shoes",      "shoes",      _ZONE_WEIGHTS["shoes"])
    apply("head",       "hat",        _ZONE_WEIGHTS["hat"])

    acc_attrs = attributes.get("accessories", [])
    if acc_attrs:
        acc_s, acc_matched = _score_accessories(detection, acc_attrs)
        z_scores["accessories"] = acc_s
        weighted    += acc_s * _ZONE_WEIGHTS["accessories"]
        weight_used += _ZONE_WEIGHTS["accessories"]
    else:
        acc_matched = []

    overall = (weighted / weight_used) if weight_used > 0 else 0.0

    return MatchResult(
        detection_index=detection.person_id,
        frame_num=detection.frame_num,
        timestamp=detection.timestamp,
        bbox=detection.bbox,
        crop=detection.crop,
        overall_score=overall,
        zone_scores=z_scores,
        dominant_colors=dom_colors,
        carried_objects=acc_matched
    )


def find_matches(detections: list, attributes: dict,
                 threshold: float = MATCH_THRESHOLD,
                 top_n: int = 20) -> list:
    """
    Score all detections against suspect attributes.
    Returns list of MatchResult sorted by score descending.
    """
    print(f"\n[Matcher] Scoring {len(detections)} detections...")
    results = [r for d in detections
               if (r := _match_one(d, attributes)).overall_score >= threshold]
    results.sort(key=lambda r: -r.overall_score)
    print(f"[Matcher] {len(results)} match(es) above threshold {threshold:.0%}")
    return results[:top_n]


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — VISUALIZER
#  Builds annotated result grid + saves crops + JSON report
# ══════════════════════════════════════════════════════════════

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_C_HIGH = (50, 200, 50)
_C_MED  = (50, 180, 220)
_C_LOW  = (50, 80, 220)


def _score_color(s: float) -> tuple:
    return _C_HIGH if s >= 0.6 else _C_MED if s >= 0.35 else _C_LOW


def _make_card(result: MatchResult, card_w: int = 160, card_h: int = 240) -> np.ndarray:
    card = np.full((card_h, card_w, 3), (30, 30, 30), dtype=np.uint8)
    img_h = card_h - 70

    crop = result.crop
    if crop is not None and crop.size > 0:
        h, w = crop.shape[:2]
        scale = min((card_w - 4) / w, img_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        xo = (card_w - 4 - nw) // 2 + 2
        yo = (img_h - nh) // 2 + 2
        card[yo:yo+nh, xo:xo+nw] = resized

    py = img_h + 2
    card[py:, :] = (20, 20, 20)
    color = _score_color(result.overall_score)

    cv2.putText(card, f"Match: {result.overall_score:.0%}", (6, py+16),
                _FONT, 0.38, color, 1, cv2.LINE_AA)

    # Score bar
    bx, bw = int(card_w * 0.075), int(card_w * 0.85)
    cv2.rectangle(card, (bx, py+22), (bx+bw, py+28), (60,60,60), -1)
    filled = int(bw * min(result.overall_score, 1.0))
    if filled > 0:
        cv2.rectangle(card, (bx, py+22), (bx+filled, py+28), color, -1)

    ts = str(timedelta(seconds=int(result.timestamp)))
    cv2.putText(card, f"Frame: {result.frame_num}", (6, py+40),
                _FONT, 0.32, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(card, f"Time:  {ts}",              (6, py+54),
                _FONT, 0.32, (160,160,160), 1, cv2.LINE_AA)

    zone_text = " ".join(f"{k[:3].upper()}:{v:.0%}"
                         for k, v in list(result.zone_scores.items())[:3])
    if zone_text:
        cv2.putText(card, zone_text, (6, py+66),
                    _FONT, 0.28, (120,120,120), 1, cv2.LINE_AA)

    cv2.rectangle(card, (0,0), (card_w-1, card_h-1), color, 2)
    return card


def build_results_grid(results: list, cols: int = 5,
                       card_w: int = 160, card_h: int = 240) -> np.ndarray:
    """Build a visual grid of all match cards. Returns numpy image."""
    if not results:
        blank = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(blank, "No matches found.", (30, 150),
                    _FONT, 0.7, (100,100,100), 1)
        return blank

    gap = 6
    rows = (len(results) + cols - 1) // cols
    grid = np.full((rows*(card_h+gap)+gap+50, cols*(card_w+gap)+gap, 3),
                   (15,15,15), dtype=np.uint8)

    cv2.putText(grid, f"Suspect Finder  |  {len(results)} possible match(es)",
                (gap, 34), _FONT, 0.55, (200,200,200), 1, cv2.LINE_AA)

    for i, r in enumerate(results):
        card = _make_card(r, card_w, card_h)
        x = gap + (i % cols) * (card_w + gap)
        y = 50 + gap + (i // cols) * (card_h + gap)
        grid[y:y+card_h, x:x+card_w] = card

    return grid


def display_results(results: list, window_title: str = "Suspect Finder",
                    save_path: str = None, cols: int = 5) -> np.ndarray:
    """
    Show results grid in an OpenCV window and/or save to file.
    Returns the grid image.
    """
    grid = build_results_grid(results, cols=cols)

    if save_path:
        cv2.imwrite(save_path, grid)
        print(f"[Visualizer] Saved: {save_path}")

    try:
        h, w = grid.shape[:2]
        max_w = 1200
        disp = cv2.resize(grid, (max_w, int(h * max_w / w))) if w > max_w else grid
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, disp)
        print("[Visualizer] Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("[Visualizer] Headless mode — window not shown.")

    return grid


def save_crops(results: list, output_dir: str) -> list:
    """Save each match crop as an individual JPEG. Returns list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, r in enumerate(results):
        if r.crop is not None and r.crop.size > 0:
            p = os.path.join(output_dir,
                             f"match_{i+1:03d}_score{r.overall_score:.0%}_frame{r.frame_num}.jpg")
            cv2.imwrite(p, r.crop)
            paths.append(p)
    print(f"[Visualizer] {len(paths)} crops saved to: {output_dir}")
    return paths


def annotate_frame(frame: np.ndarray, detections: list,
                   match_results: list) -> np.ndarray:
    """
    Draw colored bounding boxes on a frame.
    Green = high match, yellow = medium, red = low, gray = no match.
    Useful for annotating video output in your framework.
    """
    out = frame.copy()
    match_map = {(r.frame_num, r.bbox): r for r in match_results}

    for det in detections:
        r = match_map.get((det.frame_num, det.bbox))
        if r:
            color = _score_color(r.overall_score)
            label = f"Match {r.overall_score:.0%}"
            thick = 3
        else:
            color, label, thick = (80,80,80), f"{det.confidence:.0%}", 1

        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thick)
        cv2.putText(out, label, (x1, y1-8), _FONT, 0.45, color, 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — HIGH-LEVEL API
#  Clean interface for use inside your framework
# ══════════════════════════════════════════════════════════════

class SuspectFinder:
    """
    High-level interface for the Smart Surveillance Framework.

    Example:
        sf = SuspectFinder(groq_api_key="gsk_...", yolo_model="yolov8m.pt")

        # Search in video
        results = sf.search_video("cctv.mp4", "blue jacket black pants brown bag")

        # Search in single image/frame
        results = sf.search_image("frame.jpg", "red hoodie gray jeans")

        # Search directly from a numpy frame (for live/streamed video)
        results = sf.search_frame(frame_array, "white shirt blue jeans")

        for r in results:
            print(f"Score: {r.overall_score:.0%}  Frame: {r.frame_num}  Time: {r.timestamp:.1f}s")
            cv2.imshow("match", r.crop)
    """

    def __init__(self,
                 groq_api_key: str = "",
                 yolo_model: str = "yolov8m.pt",
                 yolo_confidence: float = 0.40,
                 device: str = "cpu"):
        """
        Args:
            groq_api_key:    Groq API key (free at https://console.groq.com).
                             If empty, uses rule-based parser.
            yolo_model:      Model size: yolov8n (fast) to yolov8l (accurate).
            yolo_confidence: YOLO detection confidence threshold.
            device:          'cpu' or 'cuda'.
        """
        self.api_key   = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.detector  = YOLODetector(yolo_model, yolo_confidence, device)

    def search_video(self,
                     video_path: str,
                     description: str,
                     skip_frames: int = 5,
                     threshold: float = 0.20,
                     top_n: int = 20,
                     max_frames: int = None) -> list:
        """
        Search a video file for persons matching description.

        Returns:
            List of MatchResult sorted by score (highest first).
        """
        attributes = parse_description(description, self.api_key)
        detections = self.detector.detect_video(video_path, skip_frames, max_frames)
        return find_matches(detections, attributes, threshold, top_n)

    def search_image(self,
                     image_path: str,
                     description: str,
                     threshold: float = 0.20,
                     top_n: int = 20) -> list:
        """Search a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.search_frame(frame, description, threshold, top_n)

    def search_frame(self,
                     frame: np.ndarray,
                     description: str,
                     threshold: float = 0.20,
                     top_n: int = 20) -> list:
        """
        Search a single numpy frame (BGR). Useful for live video streams.
        Note: parse_description is called each time — cache attributes if calling
              this in a loop for performance.
        """
        attributes = parse_description(description, self.api_key)
        detections = self.detector.detect_frame(frame, frame_num=0, timestamp=0.0)
        return find_matches(detections, attributes, threshold, top_n)

    def search_frame_with_attributes(self,
                                     frame: np.ndarray,
                                     attributes: dict,
                                     threshold: float = 0.20,
                                     top_n: int = 20) -> list:
        """
        Like search_frame() but accepts pre-parsed attributes dict.
        Use this in hot loops (e.g. live stream) to avoid re-parsing every frame.

        Example:
            attrs = parse_description("blue jacket black pants", api_key)
            for frame in stream:
                results = sf.search_frame_with_attributes(frame, attrs)
        """
        detections = self.detector.detect_frame(frame, frame_num=0, timestamp=0.0)
        return find_matches(detections, attributes, threshold, top_n)


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("""
╔══════════════════════════════════════════════════╗
║          🔍  SUSPECT FINDER  v1.0               ║
║  Smart Surveillance Framework — Search Module    ║
╚══════════════════════════════════════════════════╝
""")

    # Validate config
    if not VIDEO_PATH and not IMAGE_PATH:
        print("❌ Set VIDEO_PATH or IMAGE_PATH in the CONFIG block at the top of the file.")
        sys.exit(1)
    if not DESCRIPTION.strip():
        print("❌ Set DESCRIPTION in the CONFIG block at the top of the file.")
        sys.exit(1)

    print("⚙️  Config:")
    print(f"   Source      : {VIDEO_PATH or IMAGE_PATH}")
    print(f"   Description : {DESCRIPTION}")
    print(f"   Model       : {YOLO_MODEL}")
    print(f"   Threshold   : {MATCH_THRESHOLD}")
    print(f"   API key     : {'set ✓' if GROQ_API_KEY else 'not set (using rule-based parser)'}")
    print()

    # Init
    sf = SuspectFinder(
        groq_api_key=GROQ_API_KEY,
        yolo_model=YOLO_MODEL,
        yolo_confidence=YOLO_CONFIDENCE
    )

    t0 = time.time()

    # Run
    if VIDEO_PATH:
        print(f"🎬 Processing video: {VIDEO_PATH}")
        results = sf.search_video(
            VIDEO_PATH, DESCRIPTION,
            skip_frames=SKIP_FRAMES,
            threshold=MATCH_THRESHOLD,
            top_n=TOP_N
        )
    else:
        print(f"🖼️  Processing image: {IMAGE_PATH}")
        results = sf.search_image(
            IMAGE_PATH, DESCRIPTION,
            threshold=MATCH_THRESHOLD,
            top_n=TOP_N
        )

    print(f"\n⏱️  Done in {time.time()-t0:.1f}s")

    if not results:
        print(f"\n⚠️  No matches found.")
        print(f"   Try lowering MATCH_THRESHOLD (currently {MATCH_THRESHOLD}) or rephrase DESCRIPTION.")
        sys.exit(0)

    # Print summary table
    print(f"\n🎯 {len(results)} match(es) found:")
    print("─" * 62)
    print(f"  {'#':<4} {'Score':<14} {'Frame':<10} {'Time':<12} Zones")
    print("─" * 62)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.overall_score * 10) + "░" * (10 - int(r.overall_score * 10))
        ts  = str(timedelta(seconds=int(r.timestamp)))
        zs  = " ".join(f"{k[:3].upper()}:{v:.0%}" for k, v in list(r.zone_scores.items())[:3])
        print(f"  #{i:<3} {r.overall_score:.0%} {bar}  F{r.frame_num:<8} {ts:<12} {zs}")
    print("─" * 62)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(OUTPUT_DIR, "matches_grid.jpg")

    if NO_DISPLAY:
        grid = build_results_grid(results)
        cv2.imwrite(grid_path, grid)
        print(f"\n💾 Grid saved: {grid_path}")
    else:
        display_results(results, save_path=grid_path)

    if SAVE_CROPS:
        save_crops(results, os.path.join(OUTPUT_DIR, "crops"))

    # JSON report
    report = {
        "description": DESCRIPTION,
        "total_matches": len(results),
        "matches": [
            {
                "rank": i + 1,
                "score": round(r.overall_score, 3),
                "frame": r.frame_num,
                "timestamp_sec": round(r.timestamp, 2),
                "bbox": list(r.bbox),
                "zone_scores": {k: round(v, 3) for k, v in r.zone_scores.items()}
            }
            for i, r in enumerate(results)
        ]
    }
    rp = os.path.join(OUTPUT_DIR, "report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📊 Report : {rp}")
    print(f"✅ All results saved in '{OUTPUT_DIR}/'")