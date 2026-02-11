# Spica Mini GHSE Take-Home Assignment

A mini Geographic & Human Sequencing Engine

## Solution Overview

A deterministic, hybrid sequencing engine that combines **local LLM reasoning** (Qwen 2.5 1.5B) with **Python-based deterministic math** to select 2–3 places, order them, and explain the reasoning.

**Architecture**: Python Filter → LLM Reason → Python Validate
**Language**: Python 3
**LLM**: Qwen 2.5 1.5B Instruct (GGUF, runs locally, no API keys)

---

## How to Run

```bash
# 1. Install dependencies
pip install llama-cpp-python python-dotenv

# 2. Configure model path in .env
# (The .env should be inside the spica folder)
echo "SPICA_MODEL_PATH=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf" > spica/.env

# 3. Run with sample data (input.json)
python spica/spica_llm_sequencer.py

# 4. Run with custom JSON file
python spica/spica_llm_sequencer.py path/to/your_input.json
```

---

## Hybrid Architecture|

### Pipeline

```
Input JSON
    ↓
[Python] Filter & Enrich
  - Remove places closed at start_time
  - Remove places whose duration exceeds time budget
  - Compute facts: distance, walk time, remaining open minutes
    ↓
[LLM] Reason & Select (Qwen 2.5, local, temperature=0)
  - Receives facts: crowd level, type, open hours, walk times
  - Dynamically matches user preferences to place types
  - Dynamically matches 'avoid' list to place attributes (e.g., 'crowded' → excludes 'high' crowd)
  - Selects 3 places (prefer 3), orders nearest-next
  - Writes natural-language sentences for explanations
    ↓
[Python] Validate & Fix
  - Verify all IDs exist in candidate set
  - Simulate the walk: check open-at-arrival for each place
  - Recompute total_time_minutes from Python math
  - Enforce time budget (drop places that overflow)
  - Fallback to greedy nearest-next if LLM output is invalid
    ↓
Output JSON
```

### Key Design Decision: Fully Dynamic Knowledge

1.  **Preference Matching**: Dynamically understands "coffee" ↔ cafe, "quiet" ↔ bookstore/library, "walk" ↔ park.
2.  **Avoid Logic**: Dynamically understands that "crowded" applies to places with high crowd levels.
3.  **Future-Proof**: New place types or preferences work instantly without code changes.

---

## Sample Output

```json
{
  "sequence": ["p2", "p1", "p4"],
  "total_time_minutes": 125,
  "explanation": {
    "p2": "Closest to the user's starting location, quiet, and allows for a walk.",
    "p1": "Offers coffee and a longer visit duration, satisfying the coffee preference.",
    "p4": "Quiet and allows for a longer visit, satisfying the quiet preference."
  }
}
```

*(p3 "Bar C" is excluded: crowd_level "high" matches avoid "crowded", and it opens at 18:00 which is after start_time 16:30)*

---

## Required Explanations

### 1. What constraints mattered most and why?

**Most Important:**

1. **Opening hours** — Hard constraint. We cannot visit a closed place. This must be checked both during initial filtering (is it open at start_time?) and during validation (will it still be open when we arrive after walking?).

2. **Time budget** — Hard constraint. If total walking + visit time exceeds 180 minutes, the plan is infeasible. The validator enforces this by recomputing times from Python math, never trusting the LLM's arithmetic.

3. **Avoid list** — Dynamic constraint. If the user says "avoid crowded", the LLM evaluates the candidates' attributes (like `crowd_level: "high"`) and dynamically excludes them during reasoning.

4. **Preference matching** — Soft but critical. This is the primary driver of user satisfaction. The LLM handles this dynamically, understanding semantic relationships (e.g., "quiet" relates to low-crowd bookstores even if the user didn't explicitly say "bookstore").

**Why these matter more:** Opening hours and time budget are binary pass/fail — violating them makes the plan impossible. Preference matching directly affects whether the user enjoys the experience. Distance and crowd level are secondary optimisation factors.

---

### 2. What constraints were simplified or ignored?

**Simplified:**

1. **Walking speed** — Assumed constant 5 km/h. Real speed varies by terrain, fitness, and weather.

2. **Crowd levels** — Treated as static. Real crowd levels change by time of day.

3. **Preference semantics** — The LLM handles this well for common cases but may struggle with unusual preferences (e.g., "adventure" → which place type?).

4. **Distance penalty** — Applied implicitly through nearest-next ordering, not as a scoring weight.

**Ignored:**

1. Real-time traffic and weather
2. Budget/cost constraints
3. Accessibility requirements
4. Sequential dependencies (e.g., "park before cafe" for thematic coherence)
5. Category diversity enforcement (could pick 2 cafes if both score high)

**Why?** The assignment specifies a simple, deterministic system with no external APIs. These simplifications keep the system focused and debuggable.

---

### 3. What would break if the number of places doubled?

**What breaks:**

1. **LLM prompt length** — More candidates = more tokens = slower inference. With 8+ places, we might hit the context window limit (1024 tokens). Fix: increase `n_ctx` or add a Python-side pre-scoring step to shortlist top 5-6 before sending to the LLM.

2. **Scoring ambiguity** — Many places might satisfy the same preference. The LLM's selection becomes less predictable. Fix: add explicit diversity rules.

3. **Greedy suboptimality** — With more places, the nearest-next heuristic is more likely to produce suboptimal routes (zigzagging). Fix: switch to a 2-opt or nearest-insertion heuristic.

**What still works:**

- Filtering scales linearly O(n)
- Validation scales linearly O(n)
- LLM quality is stable for 5-8 candidates

---

### 4. How would the approach change for a friend group?

**Key Changes:**

1. **Preference aggregation** — Merge all members' preferences (union), then use majority voting: places matching 3/4 members' preferences rank higher.

2. **Avoid list** — Veto power: if *anyone* avoids "crowded", exclude high-crowd places.

3. **Time budget** — Use the most restrictive (shortest available time).

4. **Explanations** — Attribute to specific members: "Cafe A matches Alice's coffee preference and Bob's quiet preference."

5. **Fairness** — Ensure at least one place matches each member's top preference. Could add a constraint: "each member gets at least one place they wanted."

```python
# Example group input
"users": [
    {"name": "Alice", "preferences": ["coffee"], "avoid": ["crowded"]},
    {"name": "Bob", "preferences": ["walk", "quiet"], "avoid": []},
]
```

---

## Explicit Limitation

**This approach does not adapt to real-time changes in crowd levels or opening hours.**

The engine treats `crowd_level` and `open_from`/`open_to` as static facts from the input JSON. If crowd levels spike mid-visit or a place closes early, the plan becomes stale. A production system would need live data feeds and plan re-generation.

---

## Optional: Mobile App Integration

### Where should this logic live?

**Server-side (recommended)**

1. **Consistency** — All users get the same plan for the same input
2. **LLM hosting** — The 1GB GGUF model shouldn't ship to mobile devices
3. **Flexibility** — Can swap models, tune prompts, or A/B test server-side
4. **Security** — User location data is processed and discarded, not stored on-device

**Hybrid approach:** Server generates the plan; client caches it for offline viewing. Client-side reordering (drag-and-drop) doesn't need the server.

### API Shape

```
POST /api/v1/plan
Content-Type: application/json

Request:  { "user": {...}, "places": [...] }
Response: { "sequence": [...], "total_time_minutes": N, "explanation": {...} }
Error:    { "error": { "code": "NO_VALID_PLACES", "message": "..." } }
```

### Mobile Constraints

| Constraint | Mitigation |
|------------|------------|
| **Latency** (~5-10s for LLM) | Show loading animation; cache recent plans |
| **Offline** | Cache last plan in AsyncStorage; show "might be outdated" badge |
| **Errors** | Retry with exponential backoff; fall back to cached plan |
| **Location drift** | Regenerate if user moves >500m from plan origin |
| **Battery** | Avoid polling; use push for live updates |

---

## Code Quality

- **Comments**: Explain *why*, not *what*
- **No hardcoded preference maps**: LLM handles all semantic matching
- **Deterministic**: `temperature=0.0` → same input always gives same output
- **Graceful degradation**: If LLM fails, greedy fallback always produces a valid plan
- **Robust parsing**: 3-strategy JSON extraction handles LLM output quirks

---
