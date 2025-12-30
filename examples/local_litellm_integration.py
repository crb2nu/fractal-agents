import os
import time
from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import LiteLLM, MockLLM
from fractal_agents.memory import FractalMemory

def main():
    print("Initializing Local Integration Test...")
    
    # 1. Setup - Point to the in-cluster service
    # Note: If running this from outside the cluster, you might need port-forwarding:
    # kubectl port-forward -n ai svc/litellm 8000:8000
    # Then LITELLM_API_BASE=http://localhost:8000/v1
    
    api_base = os.getenv("LITELLM_API_BASE", "http://litellm.ai.svc.cluster.local:8000/v1")
    api_key = os.getenv("LITELLM_API_KEY", "sk-litellm-local")
    
    print(f"Connecting to LiteLLM at {api_base}...")
    
    try:
        llm = LiteLLM(api_base=api_base, api_key=api_key)
        # Simple ping to check connection
        llm.client.models.list()
        print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Falling back to MockLLM for demonstration structure.")
        llm = MockLLM()

    memory = FractalMemory()
    
    # 2. Define a Goal that exercises routing
    goal = "Create a plan to analyze an image of a Martian landscape and reason about potential water sources."
    
    print(f"Root Goal: {goal}")
    print("-" * 50)

    # 3. Create Root Node (Starting with reasoning)
    root = FractalNode(
        goal=goal,
        llm=llm,
        memory=memory,
        max_depth=2,
        task_type="reasoning"
    )

    # 4. Run
    start_time = time.time()
    try:
        final_result = root.run()
    except Exception as e:
        print(f"Execution failed: {e}")
        return

    end_time = time.time()
    print("-" * 50)
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Final Result: {final_result[:500]}...")

if __name__ == "__main__":
    main()
