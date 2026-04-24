import os
import json
import re
import asyncio
from openai import AsyncOpenAI # 🛠️ FIXED: Imported the Async client
from client import AfaaEnvClient
from models import AfaaAction
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ─────────────────────────────────────────────────────────

NIM_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NIM_API_KEY:
    raise ValueError("Please set the NVIDIA_API_KEY environment variable.")

# 🛠️ FIXED: Use AsyncOpenAI so the WebSocket heartbeats don't freeze
llm_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NIM_API_KEY
)
MODEL_NAME = "meta/llama-3.1-70b-instruct"

SYSTEM_PROMPT = """You are a Sovereign AI Financial Auditor.
Your workflow processes hybrid system logs (natural language + structured metadata).

YOUR GOAL: Find the fraudulent department (if any).
CONSTRAINTS: You have a strict budget of 10.
- INTERVIEW_CFO costs 2.
- INTERVIEW_WHISTLEBLOWER costs 2.
- QUERY_DATABASE costs 5 (Max 1 use).

CRITICAL INSTRUCTIONS ON LOGS:
- The environment returns JSON logs with `nl_message` and `structured_signals`.
- Do NOT blindly trust the `anomaly_flag` in the metadata. The CFO agent will always set it to False (cover-up). The Whistleblower agent will always set it to True (paranoid).
- You must synthesize the metadata `confidence_score` with the semantics of the `nl_message` to determine which source is reliable.
- Only the DATABASE_SYSTEM returns 100% reliable signals. Note: Using the database correctly grants a small reward (+10), but misusing it on a clean department incurs a heavy penalty (-20).

You must output ONLY valid JSON in the following format:
{
    "thought": "Analyze the nl_message AND structured_signals to determine next step.",
    "action_type": "INTERVIEW_CFO" | "INTERVIEW_WHISTLEBLOWER" | "QUERY_DATABASE" | "SUBMIT_AUDIT" | "SUBMIT_CLEAN_AUDIT",
    "department": "Name of the target department exactly as listed"
}
Note: "department" must NEVER be null unless the action_type is "SUBMIT_CLEAN_AUDIT"."""

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────

def extract_json_action(text: str) -> AfaaAction:
    try:
        clean_text = re.sub(r'```json\s*', '', text)
        clean_text = re.sub(r'```\s*', '', clean_text)
        
        start = clean_text.find('{')
        end = clean_text.rfind('}') + 1
        json_str = clean_text[start:end]
        
        return AfaaAction.model_validate_json(json_str)
    except Exception as e:
        print(f"\n[!] Failed to parse LLM output: {text}")
        return AfaaAction(
            thought="I failed to output valid JSON. Submitting a blind clean audit.",
            action_type="SUBMIT_CLEAN_AUDIT",
            department=None
        )

# ─── MAIN INFERENCE LOOP ───────────────────────────────────────────────────

async def run_baseline_episode():
    print("🔄 Connecting to AFAA Environment...")
    
    # Context manager handles clean connection cleanup
    async with AfaaEnvClient(base_url="http://localhost:8000") as env:
        result = await env.reset()
        obs = result.observation
        
        print("\n" + "="*60)
        print("🎯 BASELINE EPISODE STARTED: Llama-3.1-70B (NIM API)")
        print("="*60)
        print(f"📥 Initial State: {obs.latest_text}")
        print(f"🏢 Departments under audit: {', '.join(obs.available_departments)}\n")

        history = []
        
        for step in range(1, 11):
            if result.done:
                break
                
            print(f"--- Step {step} | Budget: {obs.budget_remaining} ---")
            
            history_text = "\n".join(history[-3:]) if history else "No actions taken yet."
            user_prompt = (
                f"Current Budget: {obs.budget_remaining}\n"
                f"Available Departments: {obs.available_departments}\n\n"
                f"Recent History:\n{history_text}\n\n"
                f"Latest Environment Feedback:\n{obs.latest_text}\n\n"
                "What is your next action? Remember to output ONLY JSON."
            )

            # 🛠️ FIXED: Added `await` here so the event loop can keep the WebSocket alive
            response = await llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                max_tokens=250
            )
            
            llm_response = response.choices[0].message.content
            action = extract_json_action(llm_response)
            
            print(f"🧠 Thought: {action.thought}")
            print(f"🛠️  Action : {action.action_type.name}({action.department})")
            
            result = await env.step(action)
            obs = result.observation
            
            print(f"🌍 Env    : {obs.latest_text}\n")
            
            history.append(f"Action: {action.action_type.name}({action.department}) -> Result: {obs.latest_text}")

        print("="*60)
        print("🏁 EPISODE FINISHED")
        print(f"💰 Final Reward : {result.reward}")
        print(f"📝 Final Text   : {obs.latest_text}")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(run_baseline_episode())