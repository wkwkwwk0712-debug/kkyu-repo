"""PMCO: interpretable prototype image-feature to music-parameter mapping.

These heuristics are design hypotheses, NOT scientifically validated perceptual laws.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Optional
import numpy as np
from PIL import Image

@dataclass
class ImageFeatures:
    brightness: float
    saturation: float
    contrast: float
    complexity: float
    warmth: float

@dataclass
class MusicPlan:
    energy: float
    valence: float
    bpm: int
    genre: str
    instrument: str
    dynamics: str
    explanation: dict

def _clip01(n):
    return float(np.clip(n, 0.0, 1.0))

def extract_features(image: Image.Image | bytes) -> ImageFeatures:
    """Return five [0,1] proxies. 'complexity' is local edge magnitude, not scene motion."""
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert('RGB')
    image.thumbnail((512, 512))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = np.asarray(image.convert('HSV'), dtype=np.float32)[..., 1] / 255.0
    diff_h = np.abs(np.diff(y, axis=1)).mean() if y.shape[1] > 1 else 0.0
    diff_v = np.abs(np.diff(y, axis=0)).mean() if y.shape[0] > 1 else 0.0
    # Approximate scaling is a project hypothesis; adjust using labeled data.
    return ImageFeatures(
        brightness=_clip01(y.mean()),
        saturation=_clip01(sat.mean()),
        contrast=_clip01(y.std() / 0.35),
        complexity=_clip01((diff_h + diff_v) / 0.28),
        warmth=_clip01(0.5 + float((r - b).mean()) * 0.9),
    )

def map_music(features: ImageFeatures, energy_override: Optional[float]=None,
              weights: tuple[float,float,float]=(0.40, 0.35, 0.25)) -> MusicPlan:
    """Transparent deterministic mapping: complexity, saturation, contrast -> energy."""
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('weights must be nonnegative and sum to a positive value')
    w /= w.sum()
    contributions = {
        'complexity': float(w[0] * features.complexity),
        'saturation': float(w[1] * features.saturation),
        'contrast': float(w[2] * features.contrast),
    }
    raw = sum(contributions.values())
    energy = _clip01(raw if energy_override is None else energy_override)
    valence = _clip01(0.45 * features.brightness + 0.35 * features.warmth + 0.20 * features.saturation)
    bpm = round(55 + 95 * energy)
    if energy < 0.33:
        genre, dynamics = 'ambient / minimalist', 'soft, slow development'
        instrument = 'warm piano, airy pads' if features.warmth > 0.5 else 'felt piano, soft strings'
    elif energy < 0.67:
        genre, dynamics = 'acoustic / cinematic', 'gentle pulse, moderate intensity'
        instrument = 'acoustic guitar, strings' if features.warmth > 0.5 else 'piano, light percussion'
    else:
        genre, dynamics = 'cinematic / energetic', 'rhythmic, lively accents'
        instrument = 'bright percussion, plucked strings' if features.warmth > 0.5 else 'synths, drums'
    return MusicPlan(energy=energy, valence=valence, bpm=bpm, genre=genre,
                     instrument=instrument, dynamics=dynamics, explanation={
                         'weight_complexity': round(float(w[0]), 3),
                         'weight_saturation': round(float(w[1]), 3),
                         'weight_contrast': round(float(w[2]), 3),
                         'weighted_complexity': round(contributions['complexity'], 4),
                         'weighted_saturation': round(contributions['saturation'], 4),
                         'weighted_contrast': round(contributions['contrast'], 4),
                         'computed_energy_before_override': round(raw, 4),
                         'energy_overridden': energy_override is not None,
                     })

def build_music_prompt(plan: MusicPlan, scene_description: str='') -> str:
    scene = scene_description.strip().replace('\n', ' ')[:500]
    return (
        'Create an original instrumental music clip. No vocals, no lyrics, '
        'no imitation of copyrighted recordings or named artists. '
        f'Target tempo approximately {plan.bpm} BPM. '
        f'Energy {plan.energy:.2f}/1, emotional brightness {plan.valence:.2f}/1. '
        f'Style: {plan.genre}. Instruments: {plan.instrument}. '
        f'Dynamics: {plan.dynamics}. '
        'Express the visual atmosphere with musical texture, not literal sounds. '
        + (f'Image scene notes: {scene}.' if scene else '')
    )

def plan_record(filename: str, features: ImageFeatures, plan: MusicPlan) -> dict:
    base={'filename':filename, **asdict(features)}
    for k,v in asdict(plan).items():
        if k != 'explanation': base[k] = v
    for k,v in plan.explanation.items(): base[k] = v
    return base


# Two explicit experiment profiles; neither is validated as "better" without human labels.
WEIGHT_PROFILES = {
    "baseline": (0.40, 0.35, 0.25),
    "complexity_experiment": (0.50, 0.20, 0.30),
    "contrast_experiment": (0.35, 0.25, 0.40),
}

def map_music_profile(features: ImageFeatures, profile: str = "baseline",
                      energy_override: Optional[float] = None) -> MusicPlan:
    """Use an explicitly selected experiment profile; baseline is unchanged."""
    if profile not in WEIGHT_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose {list(WEIGHT_PROFILES)}")
    return map_music(features, energy_override=energy_override,
                     weights=WEIGHT_PROFILES[profile])
