"""Smart City catalog.

The second evaluation domain, chosen because it is the scenario both SAF and
INF-NDN evaluate on.  Having two independent domains lets us show the approach
is not tuned to one naming style -- a claim none of the three baseline papers
is in a position to make, since each evaluates on a single domain.

Structurally identical to the hospital catalog: ten metric families across five
placements, 50 advertised services, six rewordings each.
"""

from __future__ import annotations

from . import distractors
from .builder import DomainLexicon, MetricFamily, Placement, build_catalog
from .schema import NameCatalog

ROOT = "smart-city"

ROOT_SYNONYMS = (
    "smartcity",
    "urban-area",
    "city-services",
    "municipality",
    "metro-area",
    "urban-grid",
    "civic-network",
)

METRICS = (
    MetricFamily("parking", (
        "parking-availability", "free-parking-spots", "parking-occupancy",
        "available-parking", "car-park-status", "parking-space-count",
        "vacant-parking",
    )),
    MetricFamily("pollution", (
        "pollution-level", "air-pollution", "pollution-degree", "smog-index",
        "environment-status", "emission-level", "air-contamination",
    )),
    MetricFamily("traffic", (
        "vehicle-count", "traffic-flow", "car-count", "traffic-density",
        "vehicle-throughput", "road-congestion", "traffic-volume",
    )),
    MetricFamily("transit", (
        "bus-timetable", "bus-schedules", "transit-timings", "bus-arrival-times",
        "public-transport-schedule", "departure-board", "service-timetable",
    )),
    MetricFamily("streetlight", (
        "street-lighting", "lamp-post-status", "streetlamp-control",
        "public-lighting", "light-pole-state", "illumination-level",
        "street-lamp-telemetry",
    )),
    MetricFamily("waste", (
        "waste-bin-level", "garbage-fill-level", "trash-container-status",
        "refuse-bin-capacity", "bin-fullness", "waste-collection-status",
        "dumpster-level",
    )),
    MetricFamily("water", (
        "water-consumption", "water-usage", "potable-water-flow",
        "water-meter-reading", "drinking-water-supply", "water-demand",
        "hydro-consumption",
    )),
    MetricFamily("energy", (
        "energy-consumption", "power-usage", "electricity-demand",
        "grid-load", "power-consumption", "electric-meter-reading",
        "energy-draw",
    )),
    MetricFamily("noise", (
        "noise-level", "sound-pressure", "acoustic-monitoring", "decibel-reading",
        "ambient-noise", "noise-pollution", "sound-level-meter",
    )),
    MetricFamily("crowd", (
        "crowd-monitoring", "pedestrian-count", "crowd-density",
        "footfall-counter", "people-flow", "occupancy-estimate",
        "pedestrian-density",
    )),
)

PLACEMENTS = (
    Placement("district-5", "d5", "zone-north", "north", "junction-a12"),
    Placement("district-5", "d5", "zone-south", "south", "plaza-central"),
    Placement("district-9", "d9", "zone-east", "east", "sector-31"),
    Placement("district-9", "d9", "riverside", "river", "bridge-gate"),
    Placement("old-town", "oldtown", "zone-west", "west", "market-square"),
)

VERBOSE_TEMPLATES = (
    "/{root}/{group}/{sub}/services/{metric}/{instance}",
    "/{root}/realtime/{metric}/{group_short}/{instance}",
    "/{root}/{group}/sensors/{metric}/live/{instance}",
)

# Deliberate near-misses again: building automation, healthcare and industrial
# IoT talk about energy, occupancy, air quality and temperature in contexts
# semantically adjacent to several city services without being satisfiable by
# any of them.
DISTRACTOR_ROOTS = distractors.select(
    "streaming", "banking", "building", "healthcare",
    "industrial", "ecommerce", "gaming", "agriculture",
    "education", "social",
)

LEXICON = DomainLexicon(
    domain="smart-city",
    root=ROOT,
    root_synonyms=ROOT_SYNONYMS,
    metrics=METRICS,
    placements=PLACEMENTS,
    verbose_templates=VERBOSE_TEMPLATES,
    distractor_roots=DISTRACTOR_ROOTS,
)


def catalog(n_distractors: int | None = None, *, grounded: bool = False) -> NameCatalog:
    """Build the Smart City catalog (50 services, 350 resolvable names).

    ``grounded`` adds a seventh rewording per service from a published ontology.
    Coverage is better here than in the hospital domain -- Brick, SAREF,
    Haystack and SSN/SOSA between them name energy, water, lighting, air
    quality, sound, occupancy, parking and waste level -- but transit timetables
    have no class in any of the four and fall back to the invented lexicon.
    """
    return build_catalog(LEXICON, n_distractors=n_distractors, grounded=grounded)
