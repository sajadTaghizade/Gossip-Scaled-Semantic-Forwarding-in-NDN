"""Class synonyms taken from published IoT ontologies, transcribed by hand.

Section 14 has been carrying the admission that "the catalogs are generated from
a lexicon, not collected from a deployment. The rewrite families mirror how IoT
namespaces diverge, but they are still our idea of that." A reviewer put the
objection sharply: if we invented the synonyms, we may have invented synonyms
the encoder happens to be good at, and the recognition results measure our
lexicon rather than the method.

This module removes part of that objection and no more than part of it. What is
here is a manually curated table of **class names drawn from four published
open vocabularies**, so that at least some of the wordings a producer must
recognise are ones somebody else standardised:

``Brick``      Brick Schema 1.3 point and equipment classes
               (brickschema.org) -- the buildings/HVAC/sensor vocabulary, and
               the closest published ontology to the hospital domain.
``SAREF``      ETSI SAREF, the Smart Applications REFerence ontology
               (TS 103 264) and its SAREF4BLDG / SAREF4ENER extensions.
``Haystack``   Project Haystack 4 tag names and standard point markers
               (project-haystack.org).
``SSN/SOSA``   W3C/OGC Semantic Sensor Network ontology and its SOSA core
               (www.w3.org/TR/vocab-ssn/).

Two limits, stated before the table rather than after it.

*This is transcription, not a download.* The entries below were typed from the
published vocabularies' documented class and tag names. Nothing here fetches an
ontology file at build time, so the catalog stays a pure function of this
repository -- but it also means the coverage is ours to justify, and a reader
checking it against the standards is checking our transcription.

*It grounds the vocabulary and not the request.* A real client's Interest is a
phrasing, not a class name: word order, abbreviation, locality conventions and
whatever the integrator typed at three in the morning. Those ontologies say
nothing about that, and no public NDN trace exists to say it instead. So this
grounds the synonym axis and leaves the phrasing axis exactly as invented as it
was.

*Coverage is uneven, and unevenly honest.* These are building-automation and
sensing vocabularies. They cover the environmental half of the hospital domain
-- temperature, humidity, air quality, occupancy -- and most of the smart-city
domain, and they have nothing whatever to say about heart rate, blood pressure,
glucose or infusion pumps, which no building ontology models. The clinical
metric families therefore keep their invented synonyms, and any claim made from
this table has to be read as covering the subset it actually covers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

#: The four sources, by the short label used in every entry below.
STANDARDS = ("Brick", "SAREF", "Haystack", "SSN/SOSA")


@dataclass(frozen=True)
class OntologyTerm:
    """One class or tag name, and the vocabulary it was transcribed from."""

    #: The term as an NDN path component: lower case, hyphen separated.
    term: str
    #: Which published vocabulary it came from, one of :data:`STANDARDS`.
    standard: str
    #: The name as that vocabulary writes it, so the entry can be checked.
    source_name: str

    def __post_init__(self) -> None:
        if self.standard not in STANDARDS:
            raise ValueError(f"unknown standard {self.standard!r}")
        if "/" in self.term or self.term != self.term.lower():
            raise ValueError(f"term must be a lower-case path component: {self.term!r}")


def _t(term: str, standard: str, source_name: str) -> OntologyTerm:
    return OntologyTerm(term=term, standard=standard, source_name=source_name)


#: metric key -> class names for that quantity, from the four vocabularies.
#:
#: Keyed on the ``MetricFamily.key`` each domain lexicon already uses, so a
#: catalog can ask "is there a standardised way of saying this one?" and get
#: either an answer or nothing. Metric families absent from this table -- every
#: clinical one -- keep their invented synonyms and are labelled as such.
TERMS: Dict[str, Tuple[OntologyTerm, ...]] = {
    # --- environmental sensing: covered well by all four ------------------
    "temperature": (
        _t("air-temperature-sensor", "Brick", "brick:Air_Temperature_Sensor"),
        _t("zone-air-temperature-sensor", "Brick", "brick:Zone_Air_Temperature_Sensor"),
        _t("temperature-sensor", "SAREF", "saref:TemperatureSensor"),
        _t("air-temp", "Haystack", "air temp point"),
        _t("temperature-observation", "SSN/SOSA", "sosa:Observation of Temperature"),
    ),
    "humidity": (
        _t("relative-humidity-sensor", "Brick", "brick:Relative_Humidity_Sensor"),
        _t("humidity-sensor", "SAREF", "saref:HumiditySensor"),
        _t("air-humidity", "Haystack", "air humidity point"),
        _t("humidity-observation", "SSN/SOSA", "sosa:Observation of Humidity"),
    ),
    "air-quality": (
        _t("co2-sensor", "Brick", "brick:CO2_Sensor"),
        _t("air-quality-sensor", "SAREF", "saref:Sensor measuring AirQuality"),
        _t("air-co2-concentration", "Haystack", "air co2 concentration sensor"),
        _t("particulate-matter-sensor", "Brick", "brick:PM2.5_Sensor"),
    ),
    "occupancy": (
        _t("occupancy-sensor", "Brick", "brick:Occupancy_Sensor"),
        _t("occupancy-count-sensor", "Brick", "brick:Occupancy_Count_Sensor"),
        _t("occupancy-detector", "SAREF", "saref:OccupancySensor"),
        _t("zone-occupied", "Haystack", "zone occupied point"),
    ),
    # --- smart city: partly covered ---------------------------------------
    "energy": (
        _t("electrical-power-sensor", "Brick", "brick:Electrical_Power_Sensor"),
        _t("energy-usage-sensor", "Brick", "brick:Energy_Usage_Sensor"),
        _t("power-meter", "SAREF", "saref4ener:PowerMeter"),
        _t("elec-meter-power", "Haystack", "elec meter power point"),
        _t("energy-observation", "SSN/SOSA", "sosa:Observation of EnergyConsumption"),
    ),
    "water": (
        _t("water-flow-sensor", "Brick", "brick:Water_Flow_Sensor"),
        _t("water-usage-sensor", "Brick", "brick:Water_Usage_Sensor"),
        _t("water-meter", "SAREF", "saref4watr:WaterMeter"),
        _t("water-flow", "Haystack", "water flow sensor point"),
    ),
    "streetlight": (
        _t("luminaire-driver", "Brick", "brick:Luminaire_Driver"),
        _t("lighting-system", "Brick", "brick:Lighting_System"),
        _t("light-switch", "SAREF", "saref:LightSwitch"),
        _t("lights-cmd", "Haystack", "lights cmd point"),
    ),
    "pollution": (
        _t("air-quality-sensor", "Brick", "brick:Air_Quality_Sensor"),
        _t("no2-level-sensor", "Brick", "brick:NO2_Level_Sensor"),
        _t("pollution-sensor", "SAREF", "saref:Sensor measuring Pollution"),
        _t("air-quality-observation", "SSN/SOSA", "sosa:Observation of AirQuality"),
    ),
    "noise": (
        _t("sound-sensor", "Brick", "brick:Sound_Sensor"),
        _t("noise-sensor", "SAREF", "saref:Sensor measuring Sound"),
        _t("sound-level", "Haystack", "sound level sensor point"),
        _t("sound-observation", "SSN/SOSA", "sosa:Observation of SoundPressure"),
    ),
    "crowd": (
        _t("occupancy-count-sensor", "Brick", "brick:Occupancy_Count_Sensor"),
        _t("people-count-sensor", "Brick", "brick:People_Count_Sensor"),
        _t("motion-sensor", "SAREF", "saref:MotionSensor"),
        _t("zone-occupancy", "Haystack", "zone occupancy sensor point"),
    ),
    "traffic": (
        _t("motion-sensor", "SAREF", "saref:MotionSensor"),
        _t("vehicle-count-observation", "SSN/SOSA", "sosa:Observation with a Counting procedure"),
        _t("detection-sensor", "SSN/SOSA", "ssn:System with sosa:Sensor detection"),
    ),
    "parking": (
        _t("occupancy-sensor", "Brick", "brick:Occupancy_Sensor"),
        _t("space-occupied", "Haystack", "space occupied point"),
        _t("occupancy-detector", "SAREF", "saref:OccupancySensor"),
    ),
    "waste": (
        _t("level-sensor", "Brick", "brick:Level_Sensor"),
        _t("fill-level-sensor", "SAREF", "saref:Sensor measuring Level"),
        _t("level-observation", "SSN/SOSA", "sosa:Observation of Level"),
    ),
    # --- deliberately absent ----------------------------------------------
    # heart-rate, blood-pressure, oxygen, respiration, glucose, infusion,
    # transit: no published class in Brick, SAREF, Haystack or SSN/SOSA names
    # these. Building ontologies do not model clinical devices and none of the
    # four models a bus timetable. Those families keep their invented synonyms,
    # and the coverage report below says so rather than papering over it.
}


def terms_for(metric_key: str) -> Tuple[OntologyTerm, ...]:
    """Standardised class names for one metric family, possibly none."""
    return TERMS.get(metric_key, ())


def pick(metric_key: str, offset: int) -> Optional[OntologyTerm]:
    """One standardised term by deterministic rotation, or ``None``.

    Same rotation discipline as :mod:`gsndn.datasets.builder`: distinct services
    in a family get distinct wordings, and the whole catalog stays a pure
    function of this table.
    """
    options = terms_for(metric_key)
    if not options:
        return None
    return options[offset % len(options)]


def coverage(metric_keys: Sequence[str]) -> Dict[str, object]:
    """Which of a domain's metric families this table can speak for.

    Reported into catalog metadata so that any result computed on a grounded
    catalog carries its own statement of how much of it is actually grounded.
    """
    grounded = [k for k in metric_keys if k in TERMS]
    per_standard: Dict[str, int] = {s: 0 for s in STANDARDS}
    for key in grounded:
        for term in TERMS[key]:
            per_standard[term.standard] += 1
    return {
        "metric_families": len(metric_keys),
        "grounded_families": len(grounded),
        "grounded_keys": sorted(grounded),
        "ungrounded_keys": sorted(k for k in metric_keys if k not in TERMS),
        "terms_per_standard": per_standard,
    }
