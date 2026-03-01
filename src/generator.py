import random

def generate():
    shapes = []
    for _ in range(50):
        shapes.append({
            "x": random.random() * 500,
            "y": random.random() * 500,
            "r": random.random() * 50
        })
    return shapes
