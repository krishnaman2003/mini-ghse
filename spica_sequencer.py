"""
Spica Mini GHSE Take-Home Assignment
A simple, deterministic engine for sequencing location visits based on constraints.

Author: Your Name
Approach: Filter → Score → Select → Sequence
"""

import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


# ============================================================================
# DISTANCE CALCULATION
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Returns distance in kilometers.
    Uses the Haversine formula for accuracy over short distances.
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def calculate_travel_time(distance_km: float) -> int:
    """
    Estimate walking time based on distance.
    
    Assumes average walking speed of 5 km/h (12 minutes per km).
    Returns time in minutes.
    """
    WALKING_SPEED_KM_PER_HOUR = 5.0
    return int((distance_km / WALKING_SPEED_KM_PER_HOUR) * 60)


# ============================================================================
# TIME PARSING AND VALIDATION
# ============================================================================

def parse_time(time_str: str) -> datetime:
    """Convert time string like '16:30' to datetime object (today's date)."""
    return datetime.strptime(time_str, "%H:%M")


def is_place_open(place: Dict, current_time: datetime) -> bool:
    """
    Check if a place is open at the given time.
    
    This is a HARD constraint - we cannot visit closed places.
    """
    open_from = parse_time(place["open_from"])
    open_to = parse_time(place["open_to"])
    
    return open_from <= current_time <= open_to


# ============================================================================
# FILTERING (Hard Constraints)
# ============================================================================

def filter_places(places: List[Dict], user: Dict, current_time: datetime) -> List[Dict]:
    """
    Filter places based on hard constraints that cannot be violated.
    
    Hard constraints:
    1. Place must be open at current_time
    2. Place's crowd level must not be in user's avoid list
    3. Place's duration must fit within remaining time budget (basic check)
    
    WHY: Hard constraints are binary - either a place works or it doesn't.
    We filter first to reduce the search space before scoring.
    """
    filtered = []
    
    for place in places:
        # Constraint 1: Must be open
        if not is_place_open(place, current_time):
            continue
        
        # Constraint 2: Avoid crowded if user wants to avoid "crowded"
        # Map crowd_level to generic terms
        if "crowded" in user.get("avoid", []) and place["crowd_level"] == "high":
            continue
        
        # Constraint 3: Must fit in time budget (basic filter)
        # We'll do more precise time checking during sequencing
        if place["avg_duration_minutes"] > user["time_available_minutes"]:
            continue
        
        filtered.append(place)
    
    return filtered


# ============================================================================
# SCORING (Soft Constraints)
# ============================================================================

def score_place(place: Dict, user: Dict, user_location: Tuple[float, float]) -> Tuple[float, str]:
    """
    Score a place based on soft preferences.
    
    Strategy: Additive scoring based on prompts rules:
    +2: matches preference
    +1: low crowd
    -2: in avoid list (handled by filter, but kept here for scoring consistency)
    -1: opens late
    -1: comparatively far (> 1km)
    
    Returns: (score, explanation_of_scoring)
    """
    score = 0.0
    reasons = []
    
    # Distance from user's starting location
    distance_km = haversine_distance(
        user["lat"], user["lng"],
        place["lat"], place["lng"]
    )
    
    # 1. Preference matching (+2)
    user_preferences = user.get("preferences", [])
    place_type = place["type"]
    
    # Map place types to preferences (dynamic matching could be improved later)
    if "coffee" in user_preferences and place_type == "cafe":
        score += 2
        reasons.append("perfect for your coffee craving")
    
    if "walk" in user_preferences and place_type == "park":
        score += 2
        reasons.append("great spot for a refreshing walk")
        
    if "quiet" in user_preferences and place["crowd_level"] == "low":
        score += 2
        reasons.append("matches your preference for a quiet environment")
    elif "quiet" in user_preferences and place["crowd_level"] == "medium":
        score += 1
        reasons.append("moderately quiet surroundings")

    # 2. Crowd level bonus (+1)
    if place["crowd_level"] == "low":
        score += 1
        # Avoid redundancy in reasoning if "quiet" already added a reason
        if not any("quiet" in r for r in reasons):
            reasons.append("currently has a low crowd level")
            
    # 3. Avoid list (-2)
    # Filter usually removes these, but we apply the penalty just in case
    if "crowded" in user.get("avoid", []) and place["crowd_level"] == "high":
        score -= 2
        reasons.append("unfortunately crowded right now")

    # 4. Opening time consideration (-1)
    start_time = parse_time(user["start_time"])
    open_from = parse_time(place["open_from"])
    if open_from > start_time:
        score -= 1
        reasons.append("opens a bit late relative to your start")
    
    # 5. Distance penalty (-1)
    if distance_km > 1.0:
        score -= 1
        reasons.append("not the closest option")
    elif distance_km < 0.3:
        reasons.append("conveniently very close to you")
    
    explanation = ", ".join(reasons) if reasons else "a solid choice for your itinerary"
    
    return score, explanation


# ============================================================================
# SELECTION
# ============================================================================

def select_places(scored_places: List[Tuple[Dict, float, str]], max_count: int = 3) -> List[Tuple[Dict, float, str]]:
    """
    Select top N places based on score.
    
    WHY SIMPLE: We just sort by score and take the top 2-3.
    In a production system, we might use more sophisticated selection
    (diversity, category balance, etc.), but clarity wins here.
    """
    # Sort by score (descending)
    sorted_places = sorted(scored_places, key=lambda x: x[1], reverse=True)
    
    # Take top 2-3 places
    # If we have fewer than max_count, take all
    return sorted_places[:min(max_count, len(sorted_places))]


# ============================================================================
# SEQUENCING (Greedy Nearest-Next)
# ============================================================================

def sequence_places(
    selected_places: List[Tuple[Dict, float, str]], 
    user: Dict
) -> Tuple[List[str], int, Dict[str, str]]:
    """
    Order places using greedy nearest-next heuristic.
    
    Algorithm:
    1. Start at user's current location
    2. Pick the nearest unvisited place that fits time budget
    3. Repeat until no more places fit or all visited
    
    WHY GREEDY: Simple, deterministic, easy to explain.
    Not optimal, but avoids backtracking and minimizes travel time.
    
    Returns: (sequence_ids, total_time, explanations)
    """
    sequence = []
    explanations = {}
    total_time = 0
    
    current_location = (user["lat"], user["lng"])
    current_time = parse_time(user["start_time"])
    time_budget = user["time_available_minutes"]
    
    # Track which places we've visited
    unvisited = list(selected_places)
    
    step_number = 1
    
    while unvisited and total_time < time_budget:
        # Find nearest place that still fits in time budget
        best_place = None
        best_distance = float('inf')
        best_index = -1
        
        for i, (place, score, score_explanation) in enumerate(unvisited):
            distance_km = haversine_distance(
                current_location[0], current_location[1],
                place["lat"], place["lng"]
            )
            
            travel_time = calculate_travel_time(distance_km)
            visit_time = place["avg_duration_minutes"]
            total_needed = total_time + travel_time + visit_time
            
            # Check if this place fits in remaining time
            if total_needed <= time_budget:
                # Check if place is still open when we'd arrive
                arrival_time = current_time + timedelta(minutes=travel_time)
                if is_place_open(place, arrival_time):
                    # Pick the nearest one
                    if distance_km < best_distance:
                        best_distance = distance_km
                        best_place = (place, score, score_explanation)
                        best_index = i
        
        # If no place fits, we're done
        if best_place is None:
            break
        
        # Add this place to sequence
        place, score, score_explanation = best_place
        sequence.append(place["id"])
        
        # Calculate actual travel time
        travel_time = calculate_travel_time(best_distance)
        
        # Generate explanation for this step
        explanation_parts = []
        
        if step_number == 1:
            explanation_parts.append(f"Chosen as your first stop")
        else:
            explanation_parts.append(f"Next closest stop")
        
        explanation_parts.append(score_explanation)
        
        # Add context about opening hours if relevant
        arrival_time = current_time + timedelta(minutes=travel_time)
        close_time = parse_time(place["open_to"])
        time_until_close = (close_time - arrival_time).total_seconds() / 60
        
        if time_until_close < 60:
            explanation_parts.append(f"best to visit now as it closes soon")
        
        # Combine into a natural sentence
        full_reason = ". ".join(explanation_parts).capitalize() + "."
        explanations[place["id"]] = full_reason
        
        # Update state
        total_time += travel_time + place["avg_duration_minutes"]
        current_location = (place["lat"], place["lng"])
        current_time += timedelta(minutes=travel_time + place["avg_duration_minutes"])
        unvisited.pop(best_index)
        step_number += 1
    
    return sequence, total_time, explanations


# ============================================================================
# MAIN ENGINE
# ============================================================================

def plan_itinerary(input_data: Dict) -> Dict:
    """
    Main sequencing engine.
    
    Pipeline:
    1. Filter places (hard constraints)
    2. Score places (soft preferences)
    3. Select top 2-3 places
    4. Sequence using greedy nearest-next
    5. Generate explanations
    
    Returns output in required format.
    """
    user = input_data["user"]
    places = input_data["places"]
    
    # Step 1: Filter
    start_time = parse_time(user["start_time"])
    filtered_places = filter_places(places, user, start_time)
    
    if not filtered_places:
        return {
            "sequence": [],
            "total_time_minutes": 0,
            "explanation": {"error": "No places match the constraints"}
        }
    
    # Step 2: Score
    user_location = (user["lat"], user["lng"])
    scored_places = []
    
    for place in filtered_places:
        score, explanation = score_place(place, user, user_location)
        scored_places.append((place, score, explanation))
    
    # Step 3: Select top 2-3
    selected_places = select_places(scored_places, max_count=3)
    
    if not selected_places:
        return {
            "sequence": [],
            "total_time_minutes": 0,
            "explanation": {"error": "No suitable places found"}
        }
    
    # Step 4: Sequence
    sequence, total_time, explanations = sequence_places(selected_places, user)
    
    # Step 5: Format output
    return {
        "sequence": sequence,
        "total_time_minutes": total_time,
        "explanation": explanations
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Sample input matching the problem statement
    sample_input = {
        "user": {
            "lat": 12.9716,
            "lng": 77.5946,
            "time_available_minutes": 180,
            "preferences": ["coffee", "walk", "quiet"],
            "avoid": ["crowded"],
            "start_time": "16:30"
        },
        "places": [
            {
                "id": "p1",
                "name": "Cafe A",
                "type": "cafe",
                "lat": 12.9721,
                "lng": 77.5950,
                "avg_duration_minutes": 45,
                "crowd_level": "medium",
                "open_from": "08:00",
                "open_to": "20:00"
            },
            {
                "id": "p2",
                "name": "Park B",
                "type": "park",
                "lat": 12.9730,
                "lng": 77.5932,
                "avg_duration_minutes": 40,
                "crowd_level": "low",
                "open_from": "05:00",
                "open_to": "19:00"
            },
            {
                "id": "p3",
                "name": "Bar C",
                "type": "bar",
                "lat": 12.9698,
                "lng": 77.5961,
                "avg_duration_minutes": 60,
                "crowd_level": "high",
                "open_from": "18:00",
                "open_to": "23:00"
            },
            {
                "id": "p4",
                "name": "Bookstore D",
                "type": "bookstore",
                "lat": 12.9705,
                "lng": 77.5928,
                "avg_duration_minutes": 30,
                "crowd_level": "low",
                "open_from": "10:00",
                "open_to": "21:00"
            }
        ]
    }
    
    # Run the engine
    result = plan_itinerary(sample_input)
    
    # Pretty print output
    print(json.dumps(result, indent=2))
