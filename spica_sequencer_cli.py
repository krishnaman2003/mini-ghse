"""
Enhanced version of Spica sequencer that can read from JSON file.
Usage: python3 spica_sequencer_cli.py sample_input.json
"""

import json
import sys
from spica_sequencer import plan_itinerary


def main():
    # Check if filename provided
    if len(sys.argv) < 2:
        print("Usage: python3 spica_sequencer_cli.py <input.json>")
        print("\nRunning with default sample input...")
        filename = "sample_input.json"
    else:
        filename = sys.argv[1]
    
    # Read input
    try:
        with open(filename, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filename}': {e}")
        sys.exit(1)
    
    # Run the engine
    result = plan_itinerary(input_data)
    
    # Output result
    print(json.dumps(result, indent=2))
    
    # Summary
    print(f"\n✓ Generated plan with {len(result['sequence'])} places")
    print(f"✓ Total time: {result['total_time_minutes']} minutes")
    print(f"✓ Time remaining: {input_data['user']['time_available_minutes'] - result['total_time_minutes']} minutes")


if __name__ == "__main__":
    main()
