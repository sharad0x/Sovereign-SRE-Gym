# server/AFAA_environment.py
import os
import random
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from openenv.core.env_server import Environment

try:
    from ..models import AfaaAction, AfaaObservation, AfaaState, AfaaActionType
except (ImportError, ValueError):
    from models import AfaaAction, AfaaObservation, AfaaState, AfaaActionType

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class AfaaEnvironment(Environment[AfaaAction, AfaaObservation, AfaaState]):

    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        try:
            super().__init__()
            
            nim_api_key = os.getenv("NVIDIA_API_KEY", "").strip('"').strip("'")
            if not nim_api_key:
                raise ValueError("NVIDIA_API_KEY is missing or empty!")

            self.llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nim_api_key
            )
            
            self.npc_model = os.getenv("NPC_MODEL", "meta/llama-3.1-8b-instruct")
            self.max_steps = 6 # 🛠️ FIXED: Aligned with budget constraints
            self._current_state = AfaaState()
            
            print(f"✅ AFAA Environment successfully initialized!")
            
        except Exception as e:
            print(f"❌ CRITICAL SERVER ERROR DURING INIT: {e}")
            raise

    @property
    def state(self) -> AfaaState:
        return self._current_state
    
    def _get_probabilistic_confidence(self, state: str) -> str:
        roll = random.random()
        if state == "truthful": return "HIGH" if roll < 0.75 else "MEDIUM"
        elif state == "deceptive": return "HIGH" if roll < 0.70 else "LOW"
        elif state == "uncertain": return "MEDIUM" if roll < 0.80 else random.choice(["LOW", "HIGH"])
        return "MEDIUM"

    def reset(self, seed=None, episode_id=None, **kwargs) -> AfaaObservation:
        all_depts = ["Sales", "Marketing", "R&D", "HR", "Legal", "Engineering", "IT", "Operations", "Procurement", "Compliance"]
        depts = random.sample(all_depts, random.randint(4, 6))
        
        archetype_roll = random.random()
        if archetype_roll < 0.30: archetype = "Paranoid Workplace"
        elif archetype_roll < 0.60: archetype = "Coordinated Cover-up"
        elif archetype_roll < 0.80: archetype = "Incompetent Accounting"
        else: archetype = "Rogue Actor"
            
        # 🛠️ FIXED: Multi-fraud loophole closed. Always exactly 1 fraud department.
        fraud_count = 1
        frauds = random.sample(depts, fraud_count)
        suspicion = {dept: 0.0 for dept in depts}
        
        self._current_state = AfaaState(
            episode_id=episode_id,
            step_count=0, budget=10, departments=depts, fraud_departments=frauds,
            archetype=archetype, wb_query_count=0, db_used=False,
            action_history=[], dept_suspicion=suspicion,
            wb_is_accurate=(random.random() < 0.65), 
            cfo_base_evasive=(random.random() < 0.30)
        )
        
        return AfaaObservation(
            budget_remaining=self._current_state.budget,
            available_departments=self._current_state.departments,
            latest_text=json.dumps({"source": "SYSTEM", "nl_message": "Audit protocol initiated."}),
            rule_violations=[], done=False, reward=0.0
            # 🛠️ FIXED: No dept_suspicion leakage
        )

    def step(self, action: AfaaAction, timeout_s=None, **kwargs) -> AfaaObservation:
        self._current_state.step_count += 1
        reward = 0.0
        done = False
        latest_text = ""
        dept = action.department

        # Internal belief tracking only
        for d in self._current_state.dept_suspicion:
            self._current_state.dept_suspicion[d] = max(0.0, self._current_state.dept_suspicion[d] * 0.98)

        if self._current_state.budget < 0:
            return AfaaObservation(
                budget_remaining=self._current_state.budget,
                available_departments=self._current_state.departments,
                latest_text=json.dumps({"source": "SYSTEM", "nl_message": "Budget exhausted."}),
                rule_violations=["Out of budget"], done=True, reward=-30.0
            )

        if action.action_type not in [AfaaActionType.SUBMIT_CLEAN_AUDIT] and dept not in self._current_state.departments:
             return self._fail_step(f"Invalid department: {dept}", -10.0)

        # 🛠️ FIXED: Progressive penalty removed. Replaced with constant penalty to encourage careful exploration.
        reward -= 1.0

        if action.action_type == AfaaActionType.INTERVIEW_CFO:
            self._current_state.budget -= 2
            latest_text = self._get_cfo_response(dept)
            self._update_suspicion_logic(dept, latest_text, "INTERVIEW_CFO")

        elif action.action_type == AfaaActionType.INTERVIEW_WHISTLEBLOWER:
            self._current_state.budget -= 2
            self._current_state.wb_query_count += 1
            latest_text = self._get_wb_response(dept)
            self._update_suspicion_logic(dept, latest_text, "INTERVIEW_WHISTLEBLOWER")

        elif action.action_type == AfaaActionType.QUERY_DATABASE:
            if self._current_state.db_used:
                return self._fail_step("Database protocol: Only one query permitted per audit.", -10.0)

            self._current_state.budget -= 5
            self._current_state.db_used = True
            is_fraud = dept in self._current_state.fraud_departments
            
            # 🛠️ FIXED: Database reward dominance reduced
            reward += 5.0 if is_fraud else -10.0
            
            if is_fraud:
                self._current_state.dept_suspicion[dept] = min(1.0, self._current_state.dept_suspicion[dept] + 0.5)
            else:
                self._current_state.dept_suspicion[dept] = max(0.0, self._current_state.dept_suspicion[dept] - 0.4)
            
            latest_text = json.dumps({"source": "DATABASE", "nl_message": "Scan complete.", 
                                      "structured_signals": {"anomaly": is_fraud, "confidence": "ABSOLUTE"}})

        elif action.action_type in [AfaaActionType.SUBMIT_AUDIT, AfaaActionType.SUBMIT_CLEAN_AUDIT]:
            done = True
            correct = (dept in self._current_state.fraud_departments) if action.action_type == AfaaActionType.SUBMIT_AUDIT else (not self._current_state.fraud_departments)
            reward += 100.0 if correct else -40.0

        if not done and self._current_state.step_count >= self.max_steps:
            done = True; reward -= 30.0 # Treated similarly to budget exhaustion

        return AfaaObservation(
            budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
            latest_text=latest_text, rule_violations=[], done=done, reward=reward
        )

    def _update_suspicion_logic(self, dept: str, latest_text: str, action_prefix: str):
        self._current_state.action_history.append(f"{action_prefix}({dept})")
        try:
            resp_data = json.loads(latest_text)
            conf = resp_data.get("structured_signals", {}).get("confidence", "MEDIUM")
            base_increment = 0.1 if conf == "HIGH" else 0.08
            query_count = sum(1 for h in self._current_state.action_history if f"({dept})" in h) - 1
            increment = base_increment / (1 + query_count)
            
            has_cfo = any(f"INTERVIEW_CFO({dept})" in h for h in self._current_state.action_history[:-1])
            has_wb = any(f"INTERVIEW_WHISTLEBLOWER({dept})" in h for h in self._current_state.action_history[:-1])
            if (action_prefix == "INTERVIEW_CFO" and has_wb) or (action_prefix == "INTERVIEW_WHISTLEBLOWER" and has_cfo):
                increment -= 0.03
                
            self._current_state.dept_suspicion[dept] = min(1.0, max(0.0, self._current_state.dept_suspicion[dept] + increment))
        except: pass 

    def _fail_step(self, message: str, reward: float) -> AfaaObservation:
        # 🛠️ FIXED: Invalid actions hurt exactly as much as real ones
        self._current_state.budget -= 2
        return AfaaObservation(
            budget_remaining=self._current_state.budget,
            available_departments=self._current_state.departments,
            latest_text=json.dumps({"source": "SYSTEM", "error": message}),
            rule_violations=[message], done=False, reward=reward
        )

    def _get_cfo_response(self, topic: str) -> str:
        is_fraud = topic in self._current_state.fraud_departments
        is_evasive = self._current_state.cfo_base_evasive or (self._current_state.archetype == "Incompetent Accounting")
        if is_fraud and not is_evasive:
            behavior, state = "Confident denial of known irregularities.", "deceptive"
        elif is_evasive:
            behavior, state = "Vague and guarded due to internal messy records.", "uncertain"
        else:
            behavior, state = "Cooperative and accurate reporting.", "truthful"
        conf_level = self._get_probabilistic_confidence(state)
        
        prompt = f"Role: CFO API. Topic: {topic}. Behavioral Posture: {behavior}. Output JSON object strictly with keys 'nl_message' (your response) and 'confidence' (set to '{conf_level}')."
        try:
            # 🛠️ FIXED: LLM Output Fragility & Error Masking
            res = self.llm_client.chat.completions.create(
                model=self.npc_model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            assert "confidence" in data and "nl_message" in data
            return json.dumps({"source": "CFO", "nl_message": data["nl_message"], "structured_signals": {"confidence": data["confidence"]}})
        except Exception as e:
            raise RuntimeError(f"LLM call failed (CFO): {e}")

    def _get_wb_response(self, topic: str) -> str:
        is_accurate = self._current_state.wb_is_accurate
        is_fraud = topic in self._current_state.fraud_departments
        
        # 🛠️ FIXED: Dynamic deterministic leak closure. Randomizes lie targets dynamically.
        if is_fraud and is_accurate:
            behavior, state, target = "Clear, accurate hint of fraud.", "truthful", topic
        elif not is_accurate:
            available_noise = [d for d in self._current_state.departments if d not in self._current_state.fraud_departments]
            target = random.choice(available_noise) if available_noise else topic
            behavior, state = "Confident accusation based on a misunderstanding.", "deceptive"
        else:
            behavior, state, target = "Anxious and vague about potential issues.", "uncertain", topic
            
        conf_level = self._get_probabilistic_confidence(state)
        prompt = f"Role: Whistleblower API. Topic: {topic}. Accusing: {target}. Behavioral Posture: {behavior}. Output JSON object strictly with keys 'nl_message' (your response), 'confidence' (set to '{conf_level}'), and 'target' (set to '{target}')."
        
        try:
            # 🛠️ FIXED: LLM Output Fragility & Error Masking
            res = self.llm_client.chat.completions.create(
                model=self.npc_model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            assert "confidence" in data and "nl_message" in data
            return json.dumps({"source": "WHISTLEBLOWER", "nl_message": data["nl_message"], "structured_signals": {"confidence": data["confidence"], "target": data.get("target", target)}})
        except Exception as e:
            raise RuntimeError(f"LLM call failed (Whistleblower): {e}")