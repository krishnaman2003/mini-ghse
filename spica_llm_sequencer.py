"""
Spica LLM Sequencer — Hybrid Architecture
==========================================
A deterministic sequencing engine that uses a local open-source LLM (Qwen 2.5)
for dynamic reasoning, backed by Python for all deterministic math.

Architecture:
    1. Python Fact Engine  → Haversine distances, travel times, open/close checks,
                             hard-constraint filtering (avoid list, opening hours)
    2. LLM Reasoning Engine → Dynamic preference matching, place selection,
                              nearest-next ordering, natural-language explanations
    3. Python Validator     → Recomputes total time, verifies feasibility,
                             fixes or falls back if LLM output is invalid

Why hybrid?
    - Math should never be delegated to an LLM (they hallucinate numbers).
    - Preference matching is subjective (e.g., does "quiet" relate to "bookstore"?).
      An LLM handles this dynamically without hardcoded maps.
    - Validation ensures the final output is always physically feasible.

Dependencies:
    pip install llama-cpp-python python-dotenv
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIGURATION — loaded from .env, never hardcoded
# ---------------------------------------------------------------------------

# Load environment variables from the .env file next to this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL_PATH = os.getenv("SPICA_MODEL_PATH")
if not MODEL_PATH:
    print("ERROR: SPICA_MODEL_PATH is not set in your .env file.")
    print("       Add a line like: SPICA_MODEL_PATH=/path/to/your/model.gguf")
    sys.exit(1)

# LLM inference settings — tuned for determinism and speed
LLM_TEMPERATURE = 0.0       # Fully deterministic: same input → same output
LLM_MAX_TOKENS = 512        # Enough for 3-place JSON + sentence explanations
LLM_CONTEXT_SIZE = 1024     # Compact prompt → smaller context → faster inference

# Walking speed assumption (km/h) — used only for Python-side time math
WALKING_SPEED_KMH = 5.0


# ---------------------------------------------------------------------------
# PYTHON FACT ENGINE — deterministic math, zero LLM involvement
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two coordinates in kilometres.
    Uses the Haversine formula — accurate for the short distances we deal with.
    """
    R = 6371.0  # Earth's radius in km
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def walking_minutes(distance_km: float) -> int:
    """
    Estimated walking time in whole minutes at constant speed.
    We round up to at least 1 minute — you can't teleport.
    """
    return max(1, round((distance_km / WALKING_SPEED_KMH) * 60))


def parse_time(time_str: str) -> datetime:
    """Parse an 'HH:MM' string into a datetime (today's date)."""
    return datetime.strptime(time_str, "%H:%M")


def is_open_at(place: Dict, check_time: datetime) -> bool:
    """Return True if `place` is open at `check_time`."""
    try:
        return parse_time(place["open_from"]) <= check_time <= parse_time(place["open_to"])
    except (KeyError, ValueError):
        return False


def filter_and_enrich(
    places: List[Dict], user: Dict
) -> List[Dict]:
    """
    Hard-constraint filtering + fact enrichment in one pass.

    Hard constraints (binary — violating any one disqualifies a place):
      1. Place must be open at start_time
      2. Place's duration must not exceed the total time budget

    For every surviving place, we attach computed facts:
      - _dist_km:             distance from user's start location
      - _walk_min:            walking time from start (minutes)
      - _remaining_open_min:  how many minutes until it closes

    The LLM will handle subjective/preference constraints like the 'avoid' list.
    """
    start_time = parse_time(user["start_time"])
    time_budget = user["time_available_minutes"]
    candidates = []

    for p in places:
        # Hard constraint 1: must be open at start_time
        if not is_open_at(p, start_time):
            continue

        # Hard constraint 2: visit duration alone must fit in time budget
        if p.get("avg_duration_minutes", 0) > time_budget:
            continue

        # Compute facts for this candidate
        dist = haversine_km(user["lat"], user["lng"], p["lat"], p["lng"])
        walk_min = walking_minutes(dist)

        try:
            close_dt = parse_time(p["open_to"])
            remaining = max(0, int((close_dt - start_time).total_seconds() / 60))
        except (KeyError, ValueError):
            remaining = 0

        candidates.append({
            **p,
            "_dist_km": round(dist, 3),
            "_walk_min": walk_min,
            "_remaining_open_min": remaining,
        })

    return candidates


# ---------------------------------------------------------------------------
# PROMPT CONSTRUCTION — feeds Python facts to the LLM
# ---------------------------------------------------------------------------

def build_prompt(
    user: Dict,
    candidates: List[Dict],
) -> Tuple[str, str]:
    """
    Build a system message and user message for the LLM.

    Design principles:
      1. Present pre-computed facts as a concise list (LLM must not recalculate).
      2. State user preferences and avoid list verbatim from input.
      3. Ask the LLM to reason about preference matching and things to avoid.
      4. Select 2-3 places, order them nearest-next, and write a 1-sentence explanation per place.
      5. Demand JSON-only output with an explicit schema.
    """
    prefs_str = ", ".join(user.get("preferences", []))
    avoid_str = ", ".join(user.get("avoid", [])) if user.get("avoid") else "none"

    # ----- system message -----
    system_msg = (
        "You are Spica, a place-sequencing assistant. "
        "You receive candidate places with pre-computed facts and user preferences.\n\n"
        "Rules:\n"
        "1. Select 3 places (or 2 only if time is very tight).\n"
        "2. Prioritize places matching preferences and EXCLUDE any that match the avoid list.\n"
        "3. Order them nearest-first from the user's starting location, "
        "then nearest-next from each visited place.\n"
        "4. Compute total_time_minutes = sum of walking times between consecutive places "
        "+ sum of visit durations. Use the walk_min values provided.\n"
        "5. For each place write a COMPLETE SENTENCE explaining WHY it was chosen, "
        "mentioning which user preference it satisfies. Do NOT use single words.\n"
        "6. total_time_minutes must not exceed the user's time budget.\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"sequence":["id1","id2","id3"],'
        '"total_time_minutes":<number>,'
        '"explanation":{"id1":"Full sentence reason","id2":"Full sentence reason"}}'
    )

    # ----- user message: concise fact sheet -----
    fact_lines = []
    for c in candidates:
        fact_lines.append(
            f"- {c['id']} \"{c['name']}\" | type={c['type']} | crowd={c['crowd_level']} "
            f"| visit={c['avg_duration_minutes']}min | open={c['open_from']}-{c['open_to']} "
            f"| walk_from_start={c['_walk_min']}min ({c['_dist_km']}km)"
        )

    user_msg = (
        f"User preferences: {prefs_str}\n"
        f"User avoids: {avoid_str}\n"
        f"Start time: {user['start_time']}\n"
        f"Time budget: {user['time_available_minutes']} minutes\n\n"
        f"Candidate places (all are open and pass time constraints):\n"
        + "\n".join(fact_lines)
        + "\n\nSelect 2-3 places (excluding avoided ones), order nearest-next, explain each. JSON only."
    )

    return system_msg, user_msg


# ---------------------------------------------------------------------------
# LLM INFERENCE — local Qwen 2.5 via llama-cpp-python
# ---------------------------------------------------------------------------

def run_llm(system_msg: str, user_msg: str) -> str:
    """
    Load the local GGUF model and run a single inference pass.
    Returns the raw response text.
    """
    from llama_cpp import Llama

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=LLM_CONTEXT_SIZE,
        n_gpu_layers=0,
        verbose=False,
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# JSON EXTRACTION — robust parsing of LLM output
# ---------------------------------------------------------------------------

def extract_json(raw_text: str) -> Optional[Dict]:
    """
    Extract a JSON object from LLM output.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: first { ... } blob (greedy)
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# VALIDATOR — ensures the LLM output is physically feasible
# ---------------------------------------------------------------------------

def validate_and_fix(
    raw_result: Dict,
    candidates: List[Dict],
    user: Dict,
) -> Dict:
    """
    Post-process the LLM's JSON to guarantee correctness.
    """
    valid_ids = {c["id"] for c in candidates}
    candidate_map = {c["id"]: c for c in candidates}

    # --- 1. Filter invalid IDs ---
    sequence = [pid for pid in raw_result.get("sequence", []) if pid in valid_ids]
    if not sequence:
        return _fallback_plan(candidates, user)

    # --- 2. Simulate the walk and recompute total_time ---
    current_lat, current_lng = user["lat"], user["lng"]
    current_time = parse_time(user["start_time"])
    time_budget = user["time_available_minutes"]
    validated_sequence = []
    total_time = 0

    for pid in sequence:
        place = candidate_map[pid]
        dist = haversine_km(current_lat, current_lng, place["lat"], place["lng"])
        walk = walking_minutes(dist)
        arrival = current_time + timedelta(minutes=walk)

        # Is the place still open when we'd arrive?
        if not is_open_at(place, arrival):
            continue

        visit = place["avg_duration_minutes"]
        if total_time + walk + visit > time_budget:
            continue  # Would exceed time budget — skip

        validated_sequence.append(pid)
        total_time += walk + visit
        current_lat, current_lng = place["lat"], place["lng"]
        current_time = arrival + timedelta(minutes=visit)

    if not validated_sequence:
        return _fallback_plan(candidates, user)

    # --- 3. Keep LLM explanations for validated places only ---
    explanations = raw_result.get("explanation", {})
    validated_explanations = {}
    for pid in validated_sequence:
        if pid in explanations and len(str(explanations[pid])) > 20:
            validated_explanations[pid] = explanations[pid]
        else:
            # Auto-generate a basic explanation if LLM gave a poor one
            place = candidate_map[pid]
            validated_explanations[pid] = (
                f"{place['name']} ({place['type']}) fits your time budget "
                f"and is {place['_walk_min']} min walk away."
            )

    return {
        "sequence": validated_sequence,
        "total_time_minutes": total_time,
        "explanation": validated_explanations,
    }


def _fallback_plan(
    candidates: List[Dict],
    user: Dict,
) -> Dict:
    """
    Safety net plan if the LLM output is entirely invalid.
    """
    open_candidates = [c for c in candidates if c.get("_is_open", is_open_at(c, parse_time(user["start_time"])))]
    if not open_candidates:
        return {
            "sequence": [],
            "total_time_minutes": 0,
            "explanation": {"error": "No places are currently open."},
        }

    current_lat, current_lng = user["lat"], user["lng"]
    current_time = parse_time(user["start_time"])
    budget = user["time_available_minutes"]
    seq, expls, total = [], {}, 0

    remaining = list(open_candidates)
    while remaining and len(seq) < 3:
        best_idx, best_dist = -1, float("inf")
        for i, c in enumerate(remaining):
            d = haversine_km(current_lat, current_lng, c["lat"], c["lng"])
            if d < best_dist:
                best_dist, best_idx = d, i

        c = remaining[best_idx]
        walk = walking_minutes(best_dist)
        visit = c["avg_duration_minutes"]

        if total + walk + visit > budget:
            remaining.pop(best_idx)
            continue

        arrival = current_time + timedelta(minutes=walk)
        if not is_open_at(c, arrival):
            remaining.pop(best_idx)
            continue

        seq.append(c["id"])
        expls[c["id"]] = (
            f"{c['name']} chosen as nearest open place that fits budget."
        )
        total += walk + visit
        current_lat, current_lng = c["lat"], c["lng"]
        current_time = arrival + timedelta(minutes=visit)
        remaining.pop(best_idx)

    return {
        "sequence": seq,
        "total_time_minutes": total,
        "explanation": expls,
    }


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def plan_itinerary(input_data: Dict) -> Dict:
    """
    End-to-end pipeline.
    """
    user = input_data["user"]
    places = input_data["places"]

    print("\n=== Spica LLM Sequencer ===\n")

    t0 = time.time()
    candidates = filter_and_enrich(places, user)
    
    if not candidates:
        return {
            "sequence": [],
            "total_time_minutes": 0,
            "explanation": {"error": "No places pass the hard constraints."},
        }

    system_msg, user_msg = build_prompt(user, candidates)
    
    try:
        raw_text = run_llm(system_msg, user_msg)
    except Exception as e:
        print(f"  LLM error: {e}. Falling back.")
        return _fallback_plan(candidates, user)

    raw_result = extract_json(raw_text)
    if raw_result is None:
        print("  LLM returned unparseable output. Falling back.")
        return _fallback_plan(candidates, user)

    output = validate_and_fix(raw_result, candidates, user)
    
    total_pipeline = time.time() - t0
    print(f"  Execution complete in {total_pipeline:.1f}s")
    print(f"  Places selected: {len(output['sequence'])}\n")

    return output


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Get the directory where the script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # If a JSON file path is provided as an argument, use it; 
    # otherwise default to 'input.json' in the same directory as the script
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(base_dir, "input.json")
    
    try:
        with open(filepath, "r") as f:
            input_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

    result = plan_itinerary(input_data)
    
    # Save the final result to the same directory as the script
    output_path = os.path.join(base_dir, "final_output.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(json.dumps(result, indent=2))

