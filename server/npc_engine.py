import os
import json
from openai import OpenAI

class NPCEngine:
    def __init__(self):
        nim_api_key = os.getenv("NVIDIA_API_KEY", "").strip('"').strip("'")
        if nim_api_key:
            self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nim_api_key)
            self.model = os.getenv("NPC_MODEL", "meta/llama-3.1-8b-instruct")
        else:
            self.client = None

    def render_response(self, state, topic: str, decision: dict) -> str:
        """
        Pure rendering layer. Takes a locked decision and converts it to text.
        If RL is training (fast rollout) or API fails, returns a fast fallback string.
        """
        source = decision["source"]
        target = decision["target"]
        strategy = decision["strategy"]
        conf = decision["confidence"]

        # Fast Bypass for RL Training Loops
        if getattr(state.config, "fast_rl_mode", False) or not self.client:
            return f"[{source} / {conf}] My records indicate the target is {target}."

        prompt = f"""
        Role: {source}. You are being interrogated about {topic}.
        You have ALREADY decided on the following action based on your internal game theory:
        - Strategy: {strategy}
        - Pointing to Target: {target}
        - Expressed Confidence: {conf}
        
        Write 1 to 2 sentences of dialogue reflecting this exact stance. Do not change the target or confidence.
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.3
            )
            return res.choices[0].message.content.strip()
        except Exception:
            return f"[{source} Fallback] The department in question points to {target}."