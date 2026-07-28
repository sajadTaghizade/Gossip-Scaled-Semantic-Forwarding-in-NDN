"""Smart Hospital / Internet of Healthcare Things catalog.

Ten metric families across five placements give 50 advertised services, matching
the FIB size SAF used as its default configuration.  The domain follows the
Internet of Healthcare Things scenario of SEF (Askar et al., CMC 2024) and the
smart-hospital framing of the project's own Phase 1 and Phase 2 decks.
"""

from __future__ import annotations

from . import distractors
from .builder import DomainLexicon, MetricFamily, Placement, build_catalog
from .schema import NameCatalog

ROOT = "smart-hospital"

ROOT_SYNONYMS = (
    "hospital",
    "smart-hosp",
    "medical-center",
    "healthcare-facility",
    "med-center",
    "clinic",
    "patient-care",
)

METRICS = (
    MetricFamily("temperature", (
        "temperature", "temp-sensor", "thermal-sensor", "room-temperature",
        "ambient-temperature", "temperature-reading", "thermometer",
    )),
    MetricFamily("heart-rate", (
        "heart-rate-monitor", "heartrate", "cardiac-monitor", "pulse-rate-monitor",
        "hr-sensor", "heartbeat-sensor", "cardiac-rhythm",
    )),
    MetricFamily("blood-pressure", (
        "blood-pressure", "bp-monitor", "bp-sensor", "blood-pressure-reading",
        "arterial-pressure", "bloodpressure-gauge", "systolic-diastolic",
    )),
    MetricFamily("oxygen", (
        "oxygen-level", "o2-sensor", "spo2-monitor", "oxygen-saturation",
        "blood-oxygen", "pulse-oximeter", "o2-saturation",
    )),
    MetricFamily("humidity", (
        "humidity", "humidity-sensor", "moisture-level", "relative-humidity",
        "air-moisture", "damp-level", "moisture-reading",
    )),
    MetricFamily("air-quality", (
        "air-quality", "aqi-monitor", "air-quality-index", "air-purity",
        "particulate-sensor", "air-pollution-level", "indoor-air-quality",
    )),
    MetricFamily("respiration", (
        "respiration-rate", "breathing-rate", "resp-rate-monitor", "breath-sensor",
        "respiratory-monitor", "breathing-monitor", "ventilation-rate",
    )),
    MetricFamily("glucose", (
        "glucose-level", "blood-sugar", "glucometer", "glucose-monitor",
        "blood-glucose", "sugar-level", "continuous-glucose-sensor",
    )),
    MetricFamily("infusion", (
        "infusion-pump", "iv-pump", "drip-controller", "medication-pump",
        "iv-infusion", "fluid-pump", "drug-delivery-pump",
    )),
    MetricFamily("occupancy", (
        "bed-occupancy", "bed-status", "occupancy-sensor", "bed-availability",
        "bed-sensor", "room-occupancy", "patient-presence",
    )),
)

PLACEMENTS = (
    Placement("building-a", "bldg-a", "floor-1", "fl-1", "room-101"),
    Placement("building-a", "bldg-a", "floor-2", "fl-2", "room-204"),
    Placement("building-b", "bldg-b", "floor-1", "fl-1", "room-311"),
    Placement("building-b", "bldg-b", "floor-3", "fl-3", "ward-7"),
    Placement("building-c", "bldg-c", "floor-2", "fl-2", "icu-bay-4"),
)

VERBOSE_TEMPLATES = (
    "/{root}/{group}/{sub}/sensors/{metric}/{instance}",
    "/{root}/monitoring/{metric}/{group_short}/{instance}",
    "/{root}/{group}/devices/{metric}/readings/{instance}",
)

# Unrelated application domains.  Weather, agriculture and building automation
# are the deliberate near-misses: their temperature, moisture and occupancy
# services sit close to the hospital ones in embedding space, so the similarity
# threshold has to actually discriminate rather than accept anything vaguely
# sensor-shaped.
DISTRACTOR_ROOTS = distractors.select(
    "streaming", "banking", "weather", "agriculture",
    "logistics", "ecommerce", "gaming", "building",
    "education", "social",
)

LEXICON = DomainLexicon(
    domain="smart-hospital",
    root=ROOT,
    root_synonyms=ROOT_SYNONYMS,
    metrics=METRICS,
    placements=PLACEMENTS,
    verbose_templates=VERBOSE_TEMPLATES,
    distractor_roots=DISTRACTOR_ROOTS,
)


def catalog(n_distractors: int | None = None, *, grounded: bool = False) -> NameCatalog:
    """Build the Smart Hospital catalog (50 services, 350 resolvable names).

    ``grounded`` adds a seventh rewording per service taken from a published
    ontology where one exists. Only the environmental families -- temperature,
    humidity, air quality, occupancy -- have one; Brick, SAREF, Haystack and
    SSN/SOSA model buildings and sensing, not clinical devices, so heart rate,
    blood pressure, oxygen, respiration, glucose and infusion fall back to the
    invented lexicon and are counted as fallbacks.
    """
    return build_catalog(LEXICON, n_distractors=n_distractors, grounded=grounded)
