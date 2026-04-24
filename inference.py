import os
import json
import re
import asyncio
from openai import AsyncOpenAI
from client import AfaaEnvClient
from models import AfaaAction
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ─── CONFIGURATION ─────────────────────────────────────────────────────────

# Initialize NIM API Client for Llama 3.1 70B Instruct
NIM_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NIM_API_KEY:
    raise ValueError("Please set the NVIDIA_API_KEY environment variable.")

# 🛠️ FIXED: Using AsyncOpenAI to prevent WebSocket timeouts
llm_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NIM_API_KEY
)
MODEL_NAME = "meta/llama-3.1-70b-instruct"

# 🛠️ POLICY UPGRADE: Hardcoded decision rules, budget awareness, and exploration limits
SYSTEM_PROMPT = """You are a Sovereign AI Financial Auditor.
Your workflow processes hybrid system logs (natural language + structured metadata).

YOUR GOAL: Find the fraudulent department (if any).

CONSTRAINTS & COSTS:
- INTERVIEW_CFO costs 2
- INTERVIEW_WHISTLEBLOWER costs 2
- QUERY_DATABASE costs 5 (Max 1 use)
- SUBMIT_AUDIT / SUBMIT_CLEAN_AUDIT costs 0

STRICT POLICY RULES (YOU MUST OBEY):
1. BUDGET AWARENESS:
   - NEVER select an action if its cost exceeds your `Current Budget`.
   - If `Current Budget` < 2, you MUST immediately select `SUBMIT_AUDIT` or `SUBMIT_CLEAN_AUDIT`.
2. SIGNAL HIERARCHY:
   - DATABASE (ABSOLUTE confidence) = Ground Truth (Highest Trust).
   - HIGH confidence = Medium Trust.
   - MEDIUM confidence = Low Trust.
   - LOW confidence = Very Low Trust.
3. DB INTERPRETATION RULE:
   - If DATABASE returns `anomaly: false` for a department, mark it as CLEAN and move to an unvisited department immediately.
4. EXPLORATION STRATEGY (AVOID TRAPS):
   - Do NOT spend more than 2 steps investigating a single department.
   - Track visited departments in your history. Prioritize unvisited departments.
   - If you receive no strong signals after 2 actions on a department, move on.
5. TERMINATION LOGIC:
   - If budget < 2 OR all departments have low suspicion after exploration, select `SUBMIT_CLEAN_AUDIT`.
   - If you find definitive proof of fraud (e.g., DATABASE anomaly: true), select `SUBMIT_AUDIT` with that department.

You must output ONLY valid JSON in the following format:
{
    "thought": "Evaluate budget, count steps per department, apply signal hierarchy, then decide.",
    "action_type": "INTERVIEW_CFO" | "INTERVIEW_WHISTLEBLOWER" | "QUERY_DATABASE" | "SUBMIT_AUDIT" | "SUBMIT_CLEAN_AUDIT",
    "department": "Name of the target department exactly as listed"
}
Note: "department" must NEVER be null unless the action_type is "SUBMIT_CLEAN_AUDIT"."""

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────

def extract_json_action(text: str) -> AfaaAction:
    """Extracts JSON from the LLM output and validates it against our Pydantic schema."""
    try:
        clean_text = re.sub(r'```json\s*', '', text)
        clean_text = re.sub(r'```\s*', '', clean_text)
        
        start = clean_text.find('{')
        end = clean_text.rfind('}') + 1
        json_str = clean_text[start:end]
        
        return AfaaAction.model_validate_json(json_str)
    except Exception as e:
        print(f"\n[!] Failed to parse LLM output: {text}")
        print(f"[!] Error: {e}")
        return AfaaAction(
            thought="I failed to output valid JSON. Submitting a blind clean audit.",
            action_type="SUBMIT_CLEAN_AUDIT",
            department=None
        )

# ─── MAIN INFERENCE LOOP ───────────────────────────────────────────────────

async def run_baseline_episode():
    print("🔄 Connecting to AFAA Environment...")
    async with AfaaEnvClient(base_url="http://localhost:8000") as env:
        
        result = await env.reset()
        obs = result.observation
        
        print("\n" + "="*60)
        print("🎯 BASELINE EPISODE STARTED: Llama-3.1-70B (NIM API)")
        print("="*60)
        print(f"📥 Initial State: {obs.latest_text}")
        print(f"🏢 Departments under audit: {', '.join(obs.available_departments)}\n")

        history = []
        
        for step in range(1, 16):
            if result.done:
                break
                
            print(f"--- Step {step} | Budget: {obs.budget_remaining} ---")
            
            # 🛠️ POLICY UPGRADE: Feed the ENTIRE history so the LLM can track the 2-step limit
            history_text = "\n".join(history) if history else "No actions taken yet."
            user_prompt = (
                f"Current Budget: {obs.budget_remaining}\n"
                f"Available Departments: {obs.available_departments}\n\n"
                f"Recent History (All steps taken):\n{history_text}\n\n"
                f"Latest Environment Feedback:\n{obs.latest_text}\n\n"
                "What is your next action? Remember to output ONLY JSON."
            )

            # Await the AsyncOpenAI call
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
            
            # 🛠️ FIXED: Context Bloat - Only pass the dense, critical semantics back to the LLM history
            history.append(json.dumps({
                "action": action.action_type.name,
                "target": action.department,
                "summary": obs.latest_text[:200]
            }))
            
        print("="*60)
        print("🏁 EPISODE FINISHED")
        print(f"💰 Final Reward : {result.reward}")
        print(f"📝 Final Text   : {obs.latest_text}")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(run_baseline_episode())