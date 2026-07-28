"""Unrelated-domain name material used to build hard negatives.

Every catalog mixes in Interest names that no advertised service can satisfy,
so that precision is measurable and the similarity threshold has something to
reject.  SAF's harder configuration did the same thing, with half the offered
catalog semantically unrelated to the FIB.

Roots are shared between domains where they are unrelated to both.  A few are
listed as *near-miss* roots and are deliberately adjacent in embedding space --
weather temperature against hospital room temperature, building-automation
occupancy against city crowd monitoring.  Those are the names that actually
exercise the threshold; a distractor set made only of obviously-foreign traffic
would make any threshold look good.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

DistractorRoot = Tuple[str, Sequence[str], Sequence[str]]

_INSTANCES_GENERIC = (
    "catalog", "recommended", "trending", "watchlist", "archive-2024",
    "session-7781", "region-emea", "batch-19", "daily-digest", "shard-3",
)

POOL: Dict[str, DistractorRoot] = {
    "streaming": (
        "video-streaming",
        ("movies", "series", "live-sports", "documentaries", "trailers"),
        _INSTANCES_GENERIC,
    ),
    "banking": (
        "online-banking",
        ("transactions", "account-balance", "loan-status", "card-services", "exchange-rate"),
        ("customer-4471", "branch-north", "monthly-report", "pending-queue", "archive-2024",
         "statement-q3", "wire-transfer", "overdraft", "fraud-review", "atm-network"),
    ),
    "ecommerce": (
        "e-commerce",
        ("product-catalog", "shopping-cart", "user-reviews", "order-history", "flash-sale"),
        ("electronics", "seller-3312", "cart-session", "returns-desk", "coupon-pool",
         "wishlist", "fulfilment-hub", "price-drop", "gift-cards", "marketplace-eu"),
    ),
    "gaming": (
        "gaming",
        ("match-lobby", "leaderboard", "player-inventory", "voice-chat", "patch-notes"),
        ("server-eu-3", "season-9", "clan-wars", "ranked-queue", "replay-archive",
         "tournament-final", "cosmetics-shop", "anti-cheat", "party-invite", "spectator-mode"),
    ),
    "logistics": (
        "logistics",
        ("shipment-tracking", "warehouse-stock", "fleet-position", "delivery-eta", "customs-clearance"),
        ("container-9981", "hub-central", "route-e5", "carrier-lot", "manifest",
         "cold-chain", "last-mile", "depot-south", "pallet-447", "reverse-logistics"),
    ),
    "education": (
        "e-learning",
        ("course-catalog", "lecture-video", "assignment-submission", "exam-schedule", "student-grades"),
        ("semester-fall", "cohort-12", "lab-section", "certificate", "discussion-forum",
         "office-hours", "reading-list", "peer-review", "transcript", "enrolment"),
    ),
    "social": (
        "social-network",
        ("news-feed", "direct-messages", "friend-suggestions", "story-uploads", "trending-hashtags"),
        ("profile-2210", "group-photography", "event-invite", "notification-queue", "media-cdn",
         "block-list", "reaction-counts", "reels", "poll-results", "verified-badge"),
    ),
    "industrial": (
        "industrial-iot",
        ("machine-vibration", "conveyor-speed", "furnace-temperature", "tool-wear", "batch-yield"),
        ("line-4", "plant-south", "shift-b", "press-12", "qa-station",
         "robot-arm-2", "coolant-loop", "die-cast-7", "maintenance-log", "scrap-rate"),
    ),
    # --- near-miss roots: semantically adjacent, still unsatisfiable ---
    "weather": (
        "weather",
        ("temperature", "humidity", "wind-speed", "precipitation", "air-pressure"),
        ("city-tehran", "station-42", "forecast-7day", "coastal-region", "historical",
         "mountain-pass", "hourly", "satellite-feed", "storm-warning", "uv-index"),
    ),
    "agriculture": (
        "agriculture",
        ("soil-moisture", "crop-health", "irrigation-schedule", "harvest-yield", "pest-detection"),
        ("field-north", "greenhouse-3", "orchard-b", "plot-22", "season-summer",
         "drip-line-9", "seed-trial", "livestock-pen", "silo-4", "agronomy-report"),
    ),
    "building": (
        "building-automation",
        ("indoor-air-quality", "hvac-setpoint", "room-occupancy", "lift-status", "energy-submeter"),
        ("tower-b", "floor-14", "conference-wing", "atrium", "server-room",
         "loading-bay", "roof-plant", "car-deck", "reception", "chiller-2"),
    ),
    "healthcare": (
        "healthcare",
        ("patient-vitals", "bed-occupancy", "medication-stock", "appointment-slots", "lab-results"),
        ("ward-3", "patient-8842", "pharmacy", "day-clinic", "triage",
         "radiology", "discharge-queue", "on-call-rota", "specimen-lot", "waiting-list"),
    ),
}

#: Roots whose vocabulary overlaps the hospital domain without being satisfiable.
HOSPITAL_NEAR_MISS = ("weather", "agriculture", "building")

#: Roots whose vocabulary overlaps the city domain without being satisfiable.
CITY_NEAR_MISS = ("building", "healthcare", "industrial")


def select(*keys: str) -> Tuple[DistractorRoot, ...]:
    """Look up distractor roots by key, preserving the given order."""
    missing = [k for k in keys if k not in POOL]
    if missing:
        raise KeyError(f"unknown distractor roots: {missing}")
    return tuple(POOL[k] for k in keys)
