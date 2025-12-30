from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import MockLLM
from fractal_agents.memory import FractalMemory
import time

def main():
    print("Initializing Fractal Agents System...")
    
    # Optional: ensure redis is running. 
    # If using the default localhost:6379, make sure `redis-server` is active.
    
    # 1. Setup
    memory = FractalMemory() # defaults to localhost
    llm = MockLLM() # uses mock responses
    
    # 2. Define Root Goal
    goal = "Design a completely new transportation system for Mars."
    
    print(f"Root Goal: {goal}")
    print("-" * 50)

    # 3. Create Root Node
    root = FractalNode(
        goal=goal,
        llm=llm,
        memory=memory,
        max_depth=2 # Limit depth for this demo
    )

    # 4. Run Recursion
    start_time = time.time()
    final_result = root.run()
    end_time = time.time()

    print("-" * 50)
    print("Mission Complete.")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Final Result: {final_result}")

if __name__ == "__main__":
    main()
