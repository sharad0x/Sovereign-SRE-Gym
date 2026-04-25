import os
import json
import re
import asyncio
from openai import AsyncOpenAI
from client import AfaaEnvClient
from models import AfaaAction, AfaaActionType
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ─── CONFIGURATION ─────────────────────────────────────────────────────────

# Initialize NIM API Client for Llama 3.1 70B Instruct
NIM_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NIM_API_KEY:
    raise ValueError("Please set the NVIDIA_API_KEY environment variable.")

# Using AsyncOpenAI to prevent WebSocket timeouts
llm_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NIM_API_KEY
)
MODEL_NAME = "meta/llama-3.1-70b-instruct"

# 🛠️ POLICY UPGRADE: Updated for the new RL-ready schema and action spaces
SYSTEM_PROMPT = """You are a Sovereign AI Financial Auditor.
Your workflow processes hybrid system logs (natural language + structured metadata).

YOUR GOAL: Find the ROOT CAUSE department of the systemic fraud. Fraud propagates through chains of unknown lengths. You must submit the final destination (Root Cause). Submitting an intermediary node or guessing blindly results in heavy penalties.

CONSTRAINTS: You have a strict budget. Manage it carefully.
Valid Strategic Actions:
- INTERVIEW_CFO (Cost: 2)
- INTERVIEW_WHISTLEBLOWER (Cost: 2)
- PRESSURE_CFO (Cost: 3)
- OFFER_LENIENCY (Cost: 3)
- VALIDATE_WHISTLEBLOWER (Cost: 3)
- QUERY_DATABASE (Cost: 5 - Use sparingly to ground your mathematical beliefs)
- SUBMIT_AUDIT (Ends investigation, use when confident)
- SUBMIT_CLEAN_AUDIT (Ends investigation, use if no fraud)

CRITICAL INSTRUCTIONS:
1. NEVER pass "None" or null for the department unless submitting a CLEAN_AUDIT. 
2. BEWARE OF NOISE: NPCs may drop red herrings. Pay close attention to their structured 'confidence' and 'strategy'.
3. DATABASE CLASSIFICATION: The database `structured_signals` explicitly returns `fraud_level` as "CLEAN", "INTERMEDIARY", or "ROOT". Use this strategically to break ties.
4. OBSERVE MASKS: Only select actions currently listed in 'Available Actions'.

You must output ONLY valid JSON in the following format:
{
    "thought": "Analyze the chain of clues, escalate queries if needed, filter noise, and decide.",
    "action_type": "ONE_OF_THE_VALID_ACTIONS",
    "department": "Name of the target department exactly as listed"
}"""
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
            thought="Invalid output corrected",
            action_type=AfaaActionType.INTERVIEW_WHISTLEBLOWER,
            department="Engineering"  # or random valid dept
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
        print(f"📥 Initial State: {obs.auxiliary_language}")
        print(f"🏢 Departments under audit: {', '.join(obs.available_departments)}\n")

        history = []
        
        for step in range(1, 31):  # Max steps aligned with env
            if result.done:
                break
                
            print(f"--- Step {step} | Budget: {obs.budget_remaining} ---")
            
            # 🛠️ FIXED: Feed the history and the separated signal/language channels
            history_text = "\n".join(history) if history else "No actions taken yet."
            signal_info = json.dumps(obs.last_signal) if obs.last_signal else "None"
            
            user_prompt = (
                f"Current Budget: {obs.budget_remaining}\n"
                f"Available Departments: {obs.available_departments}\n"
                f"Available Actions: {obs.available_actions}\n\n"
                f"Recent History (All steps taken):\n{history_text}\n\n"
                f"Latest Environment Feedback (Language): {obs.auxiliary_language}\n"
                f"Latest Environment Feedback (Structured Signal): {signal_info}\n\n"
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
                max_tokens=300
            )
            
            llm_response = response.choices[0].message.content
            action = extract_json_action(llm_response)
            
            print(f"🧠 Thought: {action.thought}")
            action_name = action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type)
            print(f"🛠️  Action : {action_name}({action.department})")
            
            result = await env.step(action)
            obs = result.observation
            
            # Output both language rendering and the structural mathematical signal
            print(f"🌍 Env (Text)   : {obs.auxiliary_language}")
            print(f"🌍 Env (Signal) : {obs.last_signal}\n")
            
            # 🛠️ FIXED: Context Bloat - Only pass the dense, critical semantics back to the LLM history
            history.append(json.dumps({
                "action": action_name,
                "target": action.department,
                "signal_received": obs.last_signal,
                "language_summary": obs.auxiliary_language[:150] if obs.auxiliary_language else ""
            }))
            
        print("="*60)
        print("🏁 EPISODE FINISHED")
        print(f"💰 Final Reward : {result.reward}")
        
        # Display rubric breakdown if available
        if hasattr(obs, 'rubric_scores') and obs.rubric_scores:
            print("📊 Rubric Breakdown:")
            for rubric, score in obs.rubric_scores.items():
                print(f"    - {rubric}: {score:.2f}")
                
        print(f"📝 Final Text   : {obs.auxiliary_language}")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(run_baseline_episode())