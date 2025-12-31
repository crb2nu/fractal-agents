import os
import time

from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import MockLLM, OpenAILLM
from fractal_agents.memory import FractalMemory


def main():
    print("Initializing Fractal Agents System...")
    
    # 1. Setup
    memory = FractalMemory()
    
    # Check if OPENAI_API_KEY is present to decide which LLM to use
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        print("Using OpenAILLM Interface")
        llm = OpenAILLM(model="gpt-3.5-turbo") # Or any other model
    else:
        print("Using MockLLM Interface (Set OPENAI_API_KEY to switch)")
        llm = MockLLM()
    
    # 2. Define Root Goal
    goal = "Design a completely new transportation system for Mars."
    
    print(f"Root Goal: {goal}")
    print("-" * 50)

    # 3. Create Root Node
    root = FractalNode(
        goal=goal,
        llm=llm,
        memory=memory,
        max_depth=2 # Limit depth for recursion
    )

    # 4. Run Recursion
    start_time = time.time()
    try:
        final_result = root.run()
    except Exception as e:
        print(f"Execution failed: {e}")
        return

    end_time = time.time()

    print("-" * 50)
    print("Mission Complete.")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Final Result: {final_result[:500]}...") # Truncate for display

if __name__ == "__main__":
    main()