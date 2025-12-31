import json
import os
from typing import Any, Dict

from fractal_agents.core import FractalNode
from fractal_agents.llm_interface import LiteLLM
from fractal_agents.memory import FractalMemory


class WorldBuilderAgent:
    def __init__(self, root_goal: str):
        self.root_goal = root_goal
        self.memory = FractalMemory()
        self.llm = LiteLLM()

        # Output paths
        self.base_dir = "../game/game/data"
        self.storylet_dir = os.path.join(self.base_dir, "storylets/fractal_gen")
        self.npc_dir = os.path.join(self.base_dir, "npcs/fractal_gen")
        self.tile_dir = os.path.join(self.base_dir, "tiles/fractal_gen")

        for d in [self.storylet_dir, self.npc_dir, self.tile_dir]:
            os.makedirs(d, exist_ok=True)

    def run(self):
        print(f"--- Starting Advanced Fractal World Building: {self.root_goal} ---")

        # Structure:
        # Depth 0: World Concept
        # Depth 1: Regions & Key Landmarks
        # Depth 2: Local Elements (NPCs, Map Tiles, Storylets)
        root = FractalNode(
            goal=f"Decompose this world-building vision into specific regions, landmarks, NPCs, and map tiles: {self.root_goal}",
            llm=self.llm,
            memory=self.memory,
            max_depth=2,
            task_type="reasoning",
        )

        root.run()

        print("\n--- Synthesis Phase: Categorizing and Exporting ---")
        self.export_recursive(root.id)

    def export_recursive(self, node_id: str):
        state = self.memory.get_node_state(node_id)
        if not state:
            return

        depth = state["depth"]
        # content = state['result']
        goal = state["goal"].lower()

        # Route leaf nodes to specific exporters based on content hints
        if depth == 2:
            if any(k in goal for k in ["npc", "character", "person", "villager"]):
                self.save_as_npc(state)
            elif any(k in goal for k in ["tile", "map", "terrain", "landscape"]):
                self.save_as_tile(state)
            else:
                self.save_as_storylet(state)

        for child_id in state["children_ids"]:
            self.export_recursive(child_id)

    def save_as_npc(self, state: Dict[str, Any]):
        prompt = (
            f"Convert this NPC description into a JSON profile: "
            f"{{'name': str, 'role': str, 'mood': str, 'dialogue_style': str, 'stats': {{'health': int, 'power': int}}}}.\n\n"
            f"Description: {state['result']}"
        )
        self._generate_and_save(prompt, self.npc_dir, f"npc_{state['id'][:8]}.json")

    def save_as_tile(self, state: Dict[str, Any]):
        prompt = (
            f"Convert this terrain/map tile description into a JSON tile config: "
            f"{{'tile_id': str, 'type': 'field|forest|ruins|water', 'description': str, 'effects': [str]}}.\n\n"
            f"Description: {state['result']}"
        )
        self._generate_and_save(prompt, self.tile_dir, f"tile_{state['id'][:8]}.json")

    def save_as_storylet(self, state: Dict[str, Any]):
        prompt = (
            f"Convert this narrative event into a valid Storylet JSON: "
            f"{{'storylet': {{'id': str, 'title': str, 'choices': [{{'text': str, 'outcome': str}}]}}}}.\n\n"
            f"Content: {state['result']}"
        )
        self._generate_and_save(prompt, self.storylet_dir, f"storylet_{state['id'][:8]}.json")

    def _generate_and_save(self, prompt: str, target_dir: str, filename: str):
        print(f"Generating specialized asset: {filename}...")
        raw_json = self.llm.generate_response(prompt, model_hint="general")
        try:
            start = raw_json.find("{")
            end = raw_json.rfind("}") + 1
            data = json.loads(raw_json[start:end])
            with open(os.path.join(target_dir, filename), "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error parsing JSON for {filename}: {e}")


if __name__ == "__main__":
    builder = WorldBuilderAgent(
        "A clockwork city built on top of a giant, dying whale in a sea of mercury."
    )
    builder.run()
