# Spica Quick Start Guide (LLM + Python)

A hybrid sequencing engine that uses a local LLM (Qwen 2.5 1.5B) for reasoning and Python for deterministic math.

### Key Files
1. **spica_llm_sequencer.py** - Main script (hybrid architecture)
2. **README.md** - Full explanation of logic, constraints, and trade-offs
3. **.env** - Configuration (model path)

---

## Setup

### 1. Install Dependencies
```bash
pip install llama-cpp-python python-dotenv
```

### 2. Download Model (Qwen 2.5 1.5B Instruct GGUF)
Download the model file (approx 1GB) and place it in the `model` folder.
Update your `.env` file to point to it:

```bash
# .env content
SPICA_MODEL_PATH=d:/project/assignment/model/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## How to Run

### Run with Sample Data (input.json)
```bash
python spica/spica_llm_sequencer.py
```

### Run with Custom Input
```bash
python spica/spica_llm_sequencer.py path/to/your_input.json
```

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
