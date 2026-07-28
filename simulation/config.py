"""
═══════════════════════════════════════════════════════════════════════
  Configuration for Semantic NDN Routing Simulation
  Scenario: Smart Hospital / Smart Campus IoT
═══════════════════════════════════════════════════════════════════════
"""

# ─── Random Seed (for reproducibility) ──────────────────────────────
RANDOM_SEED = 42

# ─── Simulation Parameters ──────────────────────────────────────────
NUM_REQUESTS = 1000                 # Total Interest packets to simulate
SEMANTIC_VARIATION_RATE = 0.40      # 40% of requests use semantic variants
ZIPF_ALPHA = 1.2                    # Zipf distribution skew parameter

# ─── Network Delays (milliseconds) ─────────────────────────────────
# These model a simple topology: Consumer ↔ Edge Router ↔ Producer
LINK_DELAY_CONSUMER_ROUTER = 2.0    # ms, per direction
LINK_DELAY_ROUTER_PRODUCER = 5.0    # ms, per direction
FIB_LOOKUP_DELAY = 0.01             # ms, exact-match FIB lookup (TCAM)
EMBEDDING_COMPUTE_DELAY = 8.0       # ms, MiniLM-L6 inference time
EMBEDDING_SEARCH_DELAY = 0.5        # ms, cosine similarity search
ES_CACHE_LOOKUP_DELAY = 0.1         # ms, Embedding Store cache lookup
DATA_TRANSMISSION_DELAY = 1.0       # ms, data packet transmission
JITTER_STD_FACTOR = 0.05            # ±5% Gaussian jitter on all delays

# ─── Derived Latency Values ────────────────────────────────────────
# Base round-trip time (Consumer → Router → Producer → Router → Consumer)
BASE_RTT = (2 * LINK_DELAY_CONSUMER_ROUTER +
            2 * LINK_DELAY_ROUTER_PRODUCER +
            DATA_TRANSMISSION_DELAY)                     # = 15.0 ms

EXACT_MATCH_DELAY = BASE_RTT + FIB_LOOKUP_DELAY          # ≈ 15.01 ms
SEMANTIC_MATCH_DELAY = (BASE_RTT + FIB_LOOKUP_DELAY +
                        EMBEDDING_COMPUTE_DELAY +
                        EMBEDDING_SEARCH_DELAY)           # ≈ 23.51 ms
ES_CACHE_MATCH_DELAY = (BASE_RTT + FIB_LOOKUP_DELAY +
                        ES_CACHE_LOOKUP_DELAY)            # ≈ 15.11 ms

# ─── Semantic Matching ──────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70         # Cosine similarity threshold
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ═══════════════════════════════════════════════════════════════════
#  SMART HOSPITAL NAME DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# ─── Canonical Producer Names (registered in FIB) ──────────────────
# These are the "ground truth" names that producers register.
PRODUCER_NAMES = [
    # ── Temperature Sensors ──
    "/smart-hospital/building-a/floor-1/temperature/room-101",
    "/smart-hospital/building-a/floor-1/temperature/room-102",
    "/smart-hospital/building-a/floor-1/temperature/room-103",
    "/smart-hospital/building-a/floor-2/temperature/room-201",
    "/smart-hospital/building-a/floor-2/temperature/room-202",
    "/smart-hospital/building-b/floor-1/temperature/room-301",
    # ── Heart Rate Monitors ──
    "/smart-hospital/building-a/floor-1/heart-rate-monitor/patient-001",
    "/smart-hospital/building-a/floor-1/heart-rate-monitor/patient-002",
    "/smart-hospital/building-a/floor-2/heart-rate-monitor/patient-003",
    "/smart-hospital/building-b/floor-1/heart-rate-monitor/patient-004",
    # ── Blood Pressure Sensors ──
    "/smart-hospital/building-a/floor-1/blood-pressure/patient-001",
    "/smart-hospital/building-a/floor-2/blood-pressure/patient-002",
    "/smart-hospital/building-b/floor-1/blood-pressure/patient-003",
    # ── Environmental Sensors ──
    "/smart-hospital/building-a/floor-1/humidity/room-101",
    "/smart-hospital/building-a/floor-1/air-quality/room-102",
    "/smart-hospital/building-a/floor-2/humidity/room-201",
    "/smart-hospital/building-b/floor-1/air-quality/room-301",
    # ── Oxygen Level Monitors ──
    "/smart-hospital/building-a/floor-1/oxygen-level/patient-001",
    "/smart-hospital/building-a/floor-2/oxygen-level/patient-003",
    "/smart-hospital/building-b/floor-1/oxygen-level/patient-004",
]

# ─── Semantic Variants (what consumers actually request) ───────────
# Maps: variant_name → expected_canonical_match
# These represent real-world naming inconsistencies: abbreviations,
# synonyms, different department conventions, typos, etc.
SEMANTIC_VARIANTS = {
    # ── Temperature Variants ──
    "/hospital/temp-sensor/room-101":
        "/smart-hospital/building-a/floor-1/temperature/room-101",
    "/smart-hosp/temperature-reading/room-102":
        "/smart-hospital/building-a/floor-1/temperature/room-102",
    "/hospital/room-temperature/room-103":
        "/smart-hospital/building-a/floor-1/temperature/room-103",
    "/hospital/thermal-sensor/floor-2/room-201":
        "/smart-hospital/building-a/floor-2/temperature/room-201",
    "/smart-hospital/bldg-a/temp/room-202":
        "/smart-hospital/building-a/floor-2/temperature/room-202",
    "/hospital/bldg-b/temperature-sensor/room-301":
        "/smart-hospital/building-b/floor-1/temperature/room-301",
    # ── Heart Rate Variants ──
    "/hospital/heartrate/patient-001":
        "/smart-hospital/building-a/floor-1/heart-rate-monitor/patient-001",
    "/smart-hosp/cardiac-monitor/patient-002":
        "/smart-hospital/building-a/floor-1/heart-rate-monitor/patient-002",
    "/hospital/pulse-rate-monitor/patient-003":
        "/smart-hospital/building-a/floor-2/heart-rate-monitor/patient-003",
    "/hospital/hr-sensor/patient-004":
        "/smart-hospital/building-b/floor-1/heart-rate-monitor/patient-004",
    # ── Blood Pressure Variants ──
    "/hospital/bp-monitor/patient-001":
        "/smart-hospital/building-a/floor-1/blood-pressure/patient-001",
    "/smart-hosp/blood-pressure-reading/patient-002":
        "/smart-hospital/building-a/floor-2/blood-pressure/patient-002",
    "/hospital/bp-sensor/patient-003":
        "/smart-hospital/building-b/floor-1/blood-pressure/patient-003",
    # ── Environment Variants ──
    "/hospital/humidity-sensor/room-101":
        "/smart-hospital/building-a/floor-1/humidity/room-101",
    "/smart-hosp/air-quality-index/room-102":
        "/smart-hospital/building-a/floor-1/air-quality/room-102",
    "/hospital/moisture-level/room-201":
        "/smart-hospital/building-a/floor-2/humidity/room-201",
    "/hospital/aqi-monitor/room-301":
        "/smart-hospital/building-b/floor-1/air-quality/room-301",
    # ── Oxygen Level Variants ──
    "/hospital/o2-sensor/patient-001":
        "/smart-hospital/building-a/floor-1/oxygen-level/patient-001",
    "/smart-hosp/spo2-monitor/patient-003":
        "/smart-hospital/building-a/floor-2/oxygen-level/patient-003",
    "/hospital/oxygen-saturation/patient-004":
        "/smart-hospital/building-b/floor-1/oxygen-level/patient-004",
}
