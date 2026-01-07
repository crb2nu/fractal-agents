#!/bin/bash
# Starts the Fractal Visualizer
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uvicorn fractal_agents.visualizer.app:app --host 0.0.0.0 --port 8000 --reload
