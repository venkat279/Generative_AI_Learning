"""Small shared corpus used by every script in this project.

17 short documents across 4 topics, so clustering and retrieval demos
have an obvious "right answer" to check against.
"""

CORPUS = [
    # python (0-3)
    "Python is a popular programming language known for its readability.",
    "List comprehensions in Python provide a concise way to create lists.",
    "Virtual environments help isolate Python project dependencies.",
    "Python's GIL limits true multithreading for CPU-bound tasks.",
    # cooking (4-7)
    "Kneading dough develops gluten, giving bread its chewy texture.",
    "Searing meat at high heat locks in flavor through the Maillard reaction.",
    "A pinch of salt balances sweetness in most dessert recipes.",
    "Simmering a stock slowly extracts flavor from bones and vegetables.",
    # animals (8-12)
    "Dolphins use echolocation to navigate and hunt in murky waters.",
    "Octopuses can change the color and texture of their skin instantly.",
    "Elephants communicate over long distances using infrasound.",
    "Migratory birds rely on the Earth's magnetic field to navigate.",
    "Whales can change their location and disappear.",
    # space (13-16)
    "The James Webb Space Telescope observes the universe in infrared light.",
    "Black holes warp spacetime so intensely that not even light escapes.",
    "Mars has the largest volcano in the solar system, Olympus Mons.",
    "Saturn's rings are made mostly of ice particles and rocky debris.",
]

CATEGORIES = (
    ["python"] * 4
    + ["cooking"] * 4
    + ["animals"] * 5
    + ["space"] * 4
)

MODEL_NAME = "gemini-embedding-001"  # Gemini embedding model, truncated to 768 dims
