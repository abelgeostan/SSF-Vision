"""
suspect_finder.py  —  SSF Vision · Suspect Finder Module
=========================================================
Edit the CONFIG block below and run:  python suspect_finder.py
Dependencies:  pip install ultralytics opencv-python numpy requests
Groq free API key: https://console.groq.com
"""

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
VIDEO_PATH      = "./videos/cam4.1.mp4"
IMAGE_PATH      = ""
DESCRIPTION     = "black person, green scarf, black jacket, brown bag"
GROQ_API_KEY    = ""
YOLO_MODEL      = "yolov8n.pt"
YOLO_CONFIDENCE = 0.40
SKIP_FRAMES     = 5
MATCH_THRESHOLD = 0.70
TOP_N           = 20
OUTPUT_DIR      = "results"
SAVE_CROPS      = False
NO_DISPLAY      = False
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
# ══════════════════════════════════════════════════════════════

_COLOR_RANGES = {
    "red":    [(0, 10, 60, 40, 255), (165, 10, 60, 40, 255)],
    "orange": [(10, 10, 80, 80, 255)],
    "yellow": [(22, 10, 80, 80, 255)],
    "green":  [(60, 25, 50, 40, 255)],
    "cyan":   [(90, 10, 50, 40, 255)],
    "blue":   [(110, 20, 50, 30, 255)],
    "purple": [(135, 15, 40, 30, 255)],
    "pink":   [(160, 10, 40, 100, 255)],
    "white":  [(0, 180, 0, 180, 255)],
    "black":  [(0, 180, 0, 0, 80)],
    "gray":   [(0, 180, 0, 80, 180)],
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
#  SECTION 2 — SKIN TONE ANALYZER
#  YCrCb-based with face-first, hand-fallback, confidence gating
# ══════════════════════════════════════════════════════════════

# Skin tone definitions in YCrCb space
# Each entry: (label, display_name, Cr_min, Cr_max, Cb_min, Cb_max, Y_min)
_SKIN_TONES = [
    ("very_dark",  "black",      158, 185, 108, 130, 20),
    ("dark",       "dark brown", 150, 170, 105, 125, 40),
    ("medium",     "brown",      143, 163, 100, 120, 60),
    ("light",      "light brown",137, 157,  97, 117, 80),
    ("very_light", "white",      130, 150,  93, 113, 110),
]

# Broad YCrCb skin detection mask (catches all tones)
_SKIN_CR_MIN, _SKIN_CR_MAX = 133, 185
_SKIN_CB_MIN, _SKIN_CB_MAX = 77,  127
_SKIN_Y_MIN                = 20

# Minimum skin pixel count to trust a zone
_MIN_SKIN_PIXELS = 150

# Try to load OpenCV face detector (graceful fallback if unavailable)
_FACE_CASCADE = None
try:
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _fc = cv2.CascadeClassifier(_cascade_path)
    if not _fc.empty():
        _FACE_CASCADE = _fc
        print("[SkinTone] Haar face cascade loaded.")
except Exception:
    pass


def _get_skin_mask(ycrcb: np.ndarray) -> np.ndarray:
    """Broad skin isolation mask in YCrCb."""
    return cv2.inRange(
        ycrcb,
        np.array([_SKIN_Y_MIN,  _SKIN_CR_MIN, _SKIN_CB_MIN]),
        np.array([255,           _SKIN_CR_MAX, _SKIN_CB_MAX])
    )


def _classify_tone_from_pixels(ycrcb_pixels: np.ndarray) -> tuple:
    """
    Given an (N,3) array of YCrCb skin pixels, return (tone_key, display_name, confidence).
    Uses median Cr/Cb to find closest tone bucket.
    """
    if len(ycrcb_pixels) < _MIN_SKIN_PIXELS:
        return None, "unknown", 0.0

    median_cr = float(np.median(ycrcb_pixels[:, 1]))
    median_cb = float(np.median(ycrcb_pixels[:, 2]))

    best_key, best_name, best_dist = None, "unknown", float("inf")
    for key, name, cr_min, cr_max, cb_min, cb_max, _ in _SKIN_TONES:
        cr_center = (cr_min + cr_max) / 2
        cb_center = (cb_min + cb_max) / 2
        dist = ((median_cr - cr_center) ** 2 + (median_cb - cb_center) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key  = key
            best_name = name

    # Confidence: how many pixels actually fall in the winning bucket
    winning = next(t for t in _SKIN_TONES if t[0] == best_key)
    in_bucket = np.sum(
        (ycrcb_pixels[:, 1] >= winning[2]) & (ycrcb_pixels[:, 1] <= winning[3]) &
        (ycrcb_pixels[:, 2] >= winning[4]) & (ycrcb_pixels[:, 2] <= winning[5])
    )
    confidence = min(1.0, in_bucket / len(ycrcb_pixels) * 2.5)
    return best_key, best_name, round(confidence, 3)


def analyze_skin_tone(crop: np.ndarray) -> dict:
    """
    Analyze skin tone from a person crop.

    Strategy (in order of reliability):
      1. Detect face with Haar cascade → sample face region
      2. Fixed face zone (top 15%, center 60%) → sample
      3. Hand/wrist zones (sides of upper body bottom) → sample
      4. Neck zone → sample

    Returns:
        {
          "tone":       "dark brown",   # display name
          "tone_key":   "dark",         # internal key
          "confidence": 0.78,           # 0.0 – 1.0
          "zone_used":  "face_haar",    # which zone gave the reading
          "skin_pct":   0.42            # fraction of zone that was skin
        }
    """
    result = {"tone": "unknown", "tone_key": None,
              "confidence": 0.0, "zone_used": "none", "skin_pct": 0.0}

    if crop is None or crop.size == 0:
        return result

    h, w = crop.shape[:2]
    if h < 50 or w < 20:
        return result

    ycrcb_full = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    def _sample_zone(region, zone_name):
        """Try to get a skin tone reading from a crop region."""
        if region is None or region.size == 0 or region.shape[0] < 10:
            return None
        ycc = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb) if region.shape == crop.shape else \
              cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        mask = _get_skin_mask(ycc)
        skin_pixels = ycc[mask > 0]
        total_pixels = region.shape[0] * region.shape[1]
        skin_pct = len(skin_pixels) / max(total_pixels, 1)
        if len(skin_pixels) < _MIN_SKIN_PIXELS:
            return None
        key, name, conf = _classify_tone_from_pixels(skin_pixels)
        if key is None or conf < 0.25:
            return None
        return {"tone": name, "tone_key": key, "confidence": conf,
                "zone_used": zone_name, "skin_pct": round(skin_pct, 3)}

    # ── Zone 1: Haar face detection ───────────────────────────────────
    if _FACE_CASCADE is not None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=4, minSize=(20, 20))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = crop[fy:fy+fh, fx:fx+fw]
            r = _sample_zone(face_crop, "face_haar")
            if r:
                return r

    # ── Zone 2: Fixed face region (top 15%, center 60%) ───────────────
    fh = int(h * 0.15)
    fw_start = int(w * 0.20)
    fw_end   = int(w * 0.80)
    if fh > 10 and fw_end > fw_start:
        face_fixed = crop[0:fh, fw_start:fw_end]
        r = _sample_zone(face_fixed, "face_fixed")
        if r:
            return r

    # ── Zone 3: Neck region (15%–22%, center 50%) ─────────────────────
    neck_top = int(h * 0.15)
    neck_bot = int(h * 0.22)
    neck_l   = int(w * 0.25)
    neck_r   = int(w * 0.75)
    if neck_bot > neck_top and neck_r > neck_l:
        neck = crop[neck_top:neck_bot, neck_l:neck_r]
        r = _sample_zone(neck, "neck")
        if r:
            return r

    # ── Zone 4: Wrist/hand areas (bottom of upper body, sides) ────────
    ub_top = int(h * 0.40)
    ub_bot = int(h * 0.60)
    left_hand  = crop[ub_top:ub_bot, 0:int(w * 0.20)]
    right_hand = crop[ub_top:ub_bot, int(w * 0.80):]

    for zone, name in [(left_hand, "left_wrist"), (right_hand, "right_wrist")]:
        r = _sample_zone(zone, name)
        if r:
            return r

    # ── Zone 5: Full upper body skin sampling (last resort) ───────────
    ub = crop[int(h * 0.10):int(h * 0.55), :]
    r = _sample_zone(ub, "upper_body_broad")
    if r:
        # Lower confidence since this mixes clothing
        r["confidence"] = round(r["confidence"] * 0.6, 3)
        if r["confidence"] >= 0.20:
            return r

    return result


def skin_tone_match_score(crop: np.ndarray, target_tone: str) -> tuple:
    """
    Score 0.0–1.0 of how well the crop's skin tone matches target_tone.
    Also returns the analysis result dict for metadata.

    Returns: (score, analysis_dict)
    """
    analysis = analyze_skin_tone(crop)
    conf = analysis["confidence"]

    # If we couldn't detect skin reliably, return neutral
    if conf < 0.25 or analysis["tone_key"] is None:
        return 0.5, analysis  # neutral — don't penalize

    target = target_tone.lower().strip()

    # Normalize target aliases
    _TONE_ALIASES = {
        "black": "very_dark", "dark": "dark", "dark brown": "dark",
        "brown": "medium", "medium": "medium", "tan": "medium",
        "light brown": "light", "light": "light", "olive": "light",
        "white": "very_light", "fair": "very_light", "pale": "very_light",
        "caucasian": "very_light", "asian": "light",
    }
    target_key = _TONE_ALIASES.get(target, target)

    # Tone adjacency — adjacent tones get partial credit
    _TONE_ORDER = ["very_dark", "dark", "medium", "light", "very_light"]
    if target_key not in _TONE_ORDER:
        return 0.5, analysis  # unknown target → neutral

    detected_key = analysis["tone_key"]
    ti = _TONE_ORDER.index(target_key)
    di = _TONE_ORDER.index(detected_key) if detected_key in _TONE_ORDER else -1

    if di == -1:
        return 0.5, analysis

    diff = abs(ti - di)
    if diff == 0:
        raw_score = 1.0
    elif diff == 1:
        raw_score = 0.55  # adjacent tone — partial credit
    else:
        raw_score = 0.0

    # Scale by detection confidence — if confidence is low, pull toward neutral
    score = raw_score * conf + 0.5 * (1.0 - conf)
    return round(min(1.0, score), 3), analysis


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — DESCRIPTION PARSER
# ══════════════════════════════════════════════════════════════

_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"

_PARSER_SYSTEM = """You are a forensic description parser. Extract clothing AND physical attributes from a suspect description into strict JSON.

Output ONLY valid JSON, no explanation, no markdown:
{
  "upper_body":  {"color": "blue",  "type": "jacket"} or null,
  "lower_body":  {"color": "black", "type": "pants"}  or null,
  "shoes":       {"color": "white", "type": "sneakers"} or null,
  "hat":         {"color": "red",   "type": "cap"} or null,
  "accessories": [{"color": "brown", "type": "bag"}]   or [],
  "gender": "male" or "female" or null,
  "build":  "slim" or "heavy" or "medium" or null,
  "skin_tone": "black" or "dark brown" or "brown" or "light brown" or "white" or null
}
Normalize colors to: red orange yellow green blue purple pink white black gray brown beige khaki navy maroon
Normalize skin_tone to: black dark brown brown light brown white"""


def _parse_groq(description: str, api_key: str) -> dict:
    resp = requests.post(
        _GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
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

    # Skin tone keywords
    TONE_KEYWORDS = {
        "black":       ["black person", "black man", "black woman", "african",
                        "dark skin", "very dark skin"],
        "dark brown":  ["dark brown skin", "dark complexion"],
        "brown":       ["brown skin", "brown person", "south asian", "indian",
                        "hispanic", "latino", "latina", "middle eastern",
                        "arab", "mixed race", "mixed"],
        "light brown": ["light brown skin", "light skin", "east asian",
                        "asian", "chinese", "japanese", "korean", "olive skin"],
        "white":       ["white person", "white man", "white woman",
                        "caucasian", "fair skin", "pale skin", "pale"],
    }

    def find_color(keyword):
        pat = rf'(\b(?:{"|".join(COLORS)})\b)\s+(?:\w+\s+)?{keyword}'
        m = re.search(pat, desc)
        return m.group(1) if m else "unknown"

    result = {"upper_body": None, "lower_body": None, "shoes": None,
              "hat": None, "accessories": [], "gender": None, "build": None,
              "skin_tone": None}

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

    # Skin tone detection — check from most specific to least
    for tone, keywords in TONE_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            result["skin_tone"] = tone
            break

    return result


def parse_description(description: str, api_key: str = "") -> dict:
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
#  SECTION 4 — YOLO DETECTOR
# ══════════════════════════════════════════════════════════════

_PERSON_CLS = 0
_BAG_CLS = {24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase"}


@dataclass
class Detection:
    person_id:       int
    frame_num:       int
    bbox:            tuple
    confidence:      float
    crop:            np.ndarray = field(repr=False)
    carried_objects: list       = field(default_factory=list)
    timestamp:       float      = 0.0


class YOLODetector:
    def __init__(self, model_size: str = "yolov8n.pt", confidence: float = 0.4,
                 device: str = "cpu"):
        from ultralytics import YOLO
        print(f"[Detector] Loading {model_size}...")
        self.model = YOLO(model_size)
        self.confidence = confidence
        self.device = device

    def _near(self, pb, ob, ratio=0.3):
        px1, py1, px2, py2 = pb
        ox1, oy1, ox2, oy2 = ob
        pw, ph = px2 - px1, py2 - py1
        return (ox1 < px2 + pw * ratio and ox2 > px1 - pw * ratio and
                oy1 < py2 + ph * ratio and oy2 > py1 - ph * ratio)

    def detect_frame(self, frame, frame_num=0, timestamp=0.0):
        results = self.model(
            frame, conf=self.confidence,
            classes=[_PERSON_CLS] + list(_BAG_CLS.keys()),
            verbose=False, device=self.device
        )[0]
        if results.boxes is None:
            return []
        H, W = frame.shape[:2]
        persons, bags = [], []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls == _PERSON_CLS:
                persons.append({"bbox": (x1, y1, x2, y2), "conf": conf})
            elif cls in _BAG_CLS:
                bag_crop = frame[max(0,y1):y2, max(0,x1):x2].copy()
                bags.append({"bbox": (x1, y1, x2, y2), "type": _BAG_CLS[cls],
                             "conf": conf, "crop": bag_crop})
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
                person_id=pid, frame_num=frame_num, bbox=p["bbox"],
                confidence=p["conf"], crop=crop,
                carried_objects=nearby_bags, timestamp=timestamp
            ))
        return detections

    def detect_video(self, video_path, skip_frames=5, max_frames=None,
                     progress_cb=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"[Detector] {total} frames @ {fps:.1f} FPS — every {skip_frames}")
        all_dets, frame_num, processed = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
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
#  SECTION 5 — MATCHER  (now includes skin tone scoring)
# ══════════════════════════════════════════════════════════════

# Base weights — skin tone weight is dynamically added when detectable
_ZONE_WEIGHTS_BASE = {
    "upper_body":  0.35,
    "lower_body":  0.30,
    "accessories": 0.20,
    "shoes":       0.08,
    "hat":         0.07,
}
_SKIN_TONE_WEIGHT = 0.20   # Added on top when skin is detectable;
                            # other weights scaled down proportionally


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
    skin_tone_info:  dict       = field(default_factory=dict)


def _score_zone(zone_img, target_color):
    if zone_img is None or zone_img.size == 0:
        return 0.0
    if not target_color or target_color in ("unknown", "none", "null"):
        return 0.5
    return _color_match_score(zone_img, target_color)


def _score_accessories(detection, targets):
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
    zones        = _extract_zones(detection.crop)
    z_scores     = {}
    dom_colors   = {}
    weighted     = 0.0
    weight_used  = 0.0

    # ── Determine effective weights ───────────────────────────────────
    target_skin = attributes.get("skin_tone")
    has_skin_target = bool(target_skin and target_skin not in ("null", "none", ""))

    if has_skin_target:
        # Scale down clothing weights to make room for skin tone
        scale = 1.0 - _SKIN_TONE_WEIGHT
        eff_weights = {k: v * scale for k, v in _ZONE_WEIGHTS_BASE.items()}
        eff_weights["skin_tone"] = _SKIN_TONE_WEIGHT
    else:
        eff_weights = dict(_ZONE_WEIGHTS_BASE)

    # ── Clothing zone scoring ─────────────────────────────────────────
    def apply(zone_key, attr_key, weight):
        nonlocal weighted, weight_used
        attr = attributes.get(attr_key)
        if not attr:
            return
        zone  = zones.get(zone_key)
        color = attr.get("color", "unknown") if isinstance(attr, dict) else "unknown"
        s = _score_zone(zone, color)
        z_scores[attr_key]    = s
        dom_colors[attr_key]  = _get_dominant_colors(zone) if zone is not None else []
        weighted    += s * weight
        weight_used += weight

    apply("upper_body", "upper_body", eff_weights["upper_body"])
    apply("lower_body", "lower_body", eff_weights["lower_body"])
    apply("shoes",      "shoes",      eff_weights["shoes"])
    apply("head",       "hat",        eff_weights["hat"])

    acc_attrs = attributes.get("accessories", [])
    if acc_attrs:
        acc_s, acc_matched = _score_accessories(detection, acc_attrs)
        z_scores["accessories"] = acc_s
        weighted    += acc_s * eff_weights["accessories"]
        weight_used += eff_weights["accessories"]
    else:
        acc_matched = []

    # ── Skin tone scoring ─────────────────────────────────────────────
    skin_info = {}
    if has_skin_target:
        tone_score, skin_analysis = skin_tone_match_score(detection.crop, target_skin)
        skin_info = skin_analysis
        skin_info["target"] = target_skin
        skin_info["score"]  = tone_score

        if skin_analysis["confidence"] >= 0.25:
            # Confident reading — use full skin weight
            z_scores["skin_tone"] = tone_score
            weighted    += tone_score * eff_weights["skin_tone"]
            weight_used += eff_weights["skin_tone"]
            print(f"    [SkinTone] Detected: {skin_analysis['tone']} "
                  f"(conf={skin_analysis['confidence']:.2f}, "
                  f"zone={skin_analysis['zone_used']}, "
                  f"target={target_skin}, score={tone_score:.2f})")
        else:
            # Low confidence — skip skin scoring, redistribute weight
            print(f"    [SkinTone] Low confidence ({skin_analysis['confidence']:.2f}) "
                  f"via {skin_analysis['zone_used']} — skipping skin score.")
            skin_info["score"] = None  # flag as undetermined

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
        carried_objects=acc_matched,
        skin_tone_info=skin_info
    )


def find_matches(detections, attributes, threshold=MATCH_THRESHOLD, top_n=20):
    print(f"\n[Matcher] Scoring {len(detections)} detections...")
    results = [r for d in detections
               if (r := _match_one(d, attributes)).overall_score >= threshold]
    results.sort(key=lambda r: -r.overall_score)
    print(f"[Matcher] {len(results)} match(es) above threshold {threshold:.0%}")
    return results[:top_n]


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — VISUALIZER
# ══════════════════════════════════════════════════════════════

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_C_HIGH = (50, 200, 50)
_C_MED  = (50, 180, 220)
_C_LOW  = (50, 80, 220)


def _score_color(s):
    return _C_HIGH if s >= 0.6 else _C_MED if s >= 0.35 else _C_LOW


def _make_card(result: MatchResult, card_w=160, card_h=260) -> np.ndarray:
    card = np.full((card_h, card_w, 3), (30, 30, 30), dtype=np.uint8)
    img_h = card_h - 90

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

    bx, bw = int(card_w * 0.075), int(card_w * 0.85)
    cv2.rectangle(card, (bx, py+22), (bx+bw, py+28), (60,60,60), -1)
    filled = int(bw * min(result.overall_score, 1.0))
    if filled > 0:
        cv2.rectangle(card, (bx, py+22), (bx+filled, py+28), color, -1)

    ts = str(timedelta(seconds=int(result.timestamp)))
    cv2.putText(card, f"Frame: {result.frame_num}", (6, py+40),
                _FONT, 0.32, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(card, f"Time:  {ts}", (6, py+54),
                _FONT, 0.32, (160,160,160), 1, cv2.LINE_AA)

    # Skin tone line
    st = result.skin_tone_info
    if st and st.get("tone") and st["tone"] != "unknown":
        conf = st.get("confidence", 0)
        tone_text = f"Skin: {st['tone']} ({conf:.0%})"
        cv2.putText(card, tone_text, (6, py+68),
                    _FONT, 0.28, (180, 140, 100), 1, cv2.LINE_AA)
        row_offset = 14
    else:
        row_offset = 0

    zone_text = " ".join(f"{k[:3].upper()}:{v:.0%}"
                         for k, v in list(result.zone_scores.items())[:3])
    if zone_text:
        cv2.putText(card, zone_text, (6, py + 68 + row_offset),
                    _FONT, 0.28, (120,120,120), 1, cv2.LINE_AA)

    cv2.rectangle(card, (0,0), (card_w-1, card_h-1), color, 2)
    return card


def build_results_grid(results, cols=5, card_w=160, card_h=260):
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


def display_results(results, window_title="Suspect Finder", save_path=None, cols=5):
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
        print("[Visualizer] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("[Visualizer] Headless mode.")
    return grid


def save_crops(results, output_dir):
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


def annotate_frame(frame, detections, match_results):
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
#  SECTION 7 — HIGH-LEVEL API
# ══════════════════════════════════════════════════════════════

class SuspectFinder:
    """
    High-level interface for the SSF Vision Framework.

    Example:
        sf = SuspectFinder(groq_api_key="gsk_...")
        results = sf.search_video("cctv.mp4", "black person, blue jacket, black pants")
        for r in results:
            print(f"Score: {r.overall_score:.0%}  Skin: {r.skin_tone_info}")
    """

    def __init__(self, groq_api_key="", yolo_model="yolov8n.pt",
                 yolo_confidence=0.40, device="cpu"):
        self.api_key  = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.detector = YOLODetector(yolo_model, yolo_confidence, device)

    def search_video(self, video_path, description, skip_frames=5,
                     threshold=0.70, top_n=20, max_frames=None):
        attributes = parse_description(description, self.api_key)
        detections = self.detector.detect_video(video_path, skip_frames, max_frames)
        return find_matches(detections, attributes, threshold, top_n)

    def search_image(self, image_path, description, threshold=0.70, top_n=20):
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        return self.search_frame(frame, description, threshold, top_n)

    def search_frame(self, frame, description, threshold=0.70, top_n=20):
        attributes = parse_description(description, self.api_key)
        detections = self.detector.detect_frame(frame)
        return find_matches(detections, attributes, threshold, top_n)

    def search_frame_with_attributes(self, frame, attributes,
                                     threshold=0.70, top_n=20):
        detections = self.detector.detect_frame(frame)
        return find_matches(detections, attributes, threshold, top_n)


# ══════════════════════════════════════════════════════════════
#  SECTION 8 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║       🔍  SUSPECT FINDER  v2.0  (+ SkinTone)    ║
║  Smart Surveillance Framework — Search Module    ║
╚══════════════════════════════════════════════════╝
""")
    if not VIDEO_PATH and not IMAGE_PATH:
        print("❌ Set VIDEO_PATH or IMAGE_PATH in CONFIG.")
        sys.exit(1)
    if not DESCRIPTION.strip():
        print("❌ Set DESCRIPTION in CONFIG.")
        sys.exit(1)

    print(f"   Source      : {VIDEO_PATH or IMAGE_PATH}")
    print(f"   Description : {DESCRIPTION}")
    print(f"   Threshold   : {MATCH_THRESHOLD}")
    print(f"   API key     : {'set ✓' if GROQ_API_KEY else 'not set (rule-based)'}")
    print()

    sf = SuspectFinder(groq_api_key=GROQ_API_KEY, yolo_model=YOLO_MODEL,
                       yolo_confidence=YOLO_CONFIDENCE)
    t0 = time.time()

    if VIDEO_PATH:
        results = sf.search_video(VIDEO_PATH, DESCRIPTION,
                                  skip_frames=SKIP_FRAMES,
                                  threshold=MATCH_THRESHOLD, top_n=TOP_N)
    else:
        results = sf.search_image(IMAGE_PATH, DESCRIPTION,
                                  threshold=MATCH_THRESHOLD, top_n=TOP_N)

    print(f"\n⏱️  Done in {time.time()-t0:.1f}s")

    if not results:
        print(f"\n⚠️  No matches found. Try lowering MATCH_THRESHOLD ({MATCH_THRESHOLD}).")
        sys.exit(0)

    print(f"\n🎯 {len(results)} match(es):")
    print("─" * 72)
    print(f"  {'#':<4} {'Score':<14} {'Skin Tone':<16} {'Conf':<8} {'Frame':<10} Zones")
    print("─" * 72)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.overall_score * 10) + "░" * (10 - int(r.overall_score * 10))
        ts  = str(timedelta(seconds=int(r.timestamp)))
        st  = r.skin_tone_info
        tone_str = f"{st.get('tone','—'):<14}" if st else f"{'—':<14}"
        conf_str = f"{st.get('confidence',0):.0%}" if st and st.get('confidence') else "—"
        zs   = " ".join(f"{k[:3].upper()}:{v:.0%}"
                        for k, v in list(r.zone_scores.items())[:3])
        print(f"  #{i:<3} {r.overall_score:.0%} {bar}  {tone_str} {conf_str:<8} F{r.frame_num:<8} {zs}")
    print("─" * 72)

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
                "zone_scores": {k: round(v, 3) for k, v in r.zone_scores.items()},
                "skin_tone": r.skin_tone_info
            }
            for i, r in enumerate(results)
        ]
    }
    rp = os.path.join(OUTPUT_DIR, "report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📊 Report: {rp}")
    print(f"✅ Results saved in '{OUTPUT_DIR}/'")