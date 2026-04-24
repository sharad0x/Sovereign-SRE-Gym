# test_stationarity.py
import asyncio
import json
from models import AfaaAction, AfaaActionType, AfaaState 
from client import AfaaEnvClient

async def test_transition_stability():
    async with AfaaEnvClient(base_url="http://localhost:8000") as env:
        print("🔄 Resetting environment for stationarity test...")
        await env.reset()
        
        # 🛠️ FIXED: Await the state method call
        state_obj = await env.state() 
        locked_state = state_obj.model_dump()
        
        target_dept = state_obj.departments[0]
        print(f"🧪 Testing consistency for: {target_dept}")
        
        action = AfaaAction(
            thought="Test action for stationarity.", 
            action_type=AfaaActionType.INTERVIEW_WHISTLEBLOWER, 
            department=target_dept
        )
        
        # Step 1: Get first response
        res1 = await env.step(action)
        data1 = json.loads(res1.observation.latest_text)
        sig1 = data1.get("structured_signals", {})
        
        # Step 2: Get second response (Consistency check)
        # Because traits are locked in state, the behavior should remain consistent
        res2 = await env.step(action)
        data2 = json.loads(res2.observation.latest_text)
        sig2 = data2.get("structured_signals", {})
        
        # Verify Whistleblower target doesn't change randomly mid-episode
        if sig1.get("claims_department") == sig2.get("claims_department"):
            print("✅ Stationarity Verified: Whistleblower target is consistent.")
        else:
            print("❌ Stationarity Failed: Whistleblower target changed randomly.")

        if sig1.get("confidence_score") == sig2.get("confidence_score"):
             print("✅ Stationarity Verified: Confidence scores are stable.")

if __name__ == "__main__":
    try:
        asyncio.run(test_transition_stability())
    except Exception as e:
        print(f"❌ Test crashed: {e}")