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
            if not nim_api_key: raise ValueError("NVIDIA_API_KEY is missing!")
            self.llm_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nim_api_key)
            self.npc_model = os.getenv("NPC_MODEL", "meta/llama-3.1-8b-instruct")
            self.max_steps = 10 
            self._current_state = AfaaState()
        except Exception as e:
            raise RuntimeError(f"CRITICAL SERVER ERROR DURING INIT: {e}")

    @property
    def state(self) -> AfaaState: return self._current_state
    
    def _get_probabilistic_confidence(self, state: str) -> str:
        roll = random.random()
        if state == "truthful": return "HIGH" if roll < 0.75 else "MEDIUM"
        elif state == "deceptive": return "HIGH" if roll < 0.70 else "LOW"
        return "MEDIUM" if roll < 0.80 else random.choice(["LOW", "HIGH"])

    def reset(self, seed=None, episode_id=None, **kwargs) -> AfaaObservation:
        all_depts = ["Sales", "Marketing", "R&D", "HR", "Legal", "Engineering", "IT", "Operations"]
        depts = random.sample(all_depts, random.randint(5, 7))
        
        chain_length = random.randint(3, min(5, len(depts)))
        fraud_chain = random.sample(depts, chain_length) 
        
        archetype_roll = random.random()
        if archetype_roll < 0.30: archetype = "Paranoid Workplace"
        elif archetype_roll < 0.60: archetype = "Coordinated Cover-up"
        elif archetype_roll < 0.80: archetype = "Incompetent Accounting"
        else: archetype = "Rogue Actor"
            
        self._current_state = AfaaState(
            episode_id=episode_id, step_count=0, budget=20, departments=depts, 
            fraud_chain=fraud_chain, archetype=archetype, wb_query_count=0, 
            db_used=False, action_history=[], dept_suspicion={d: 0.0 for d in depts},
            wb_is_accurate=(random.random() < 0.65), cfo_base_evasive=(random.random() < 0.30)
        )
        
        return AfaaObservation(
            budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
            latest_text=json.dumps({"source": "SYSTEM", "nl_message": "Audit protocol initiated. WARNING: Fraud propagates dynamically. Find the ROOT CAUSE."}),
            rule_violations=[], done=False, reward=0.0
        )

    def step(self, action: AfaaAction, timeout_s=None, **kwargs) -> AfaaObservation:
        self._current_state.step_count += 1
        reward = -1.0 
        done = False
        latest_text = ""
        dept = action.department

        if self._current_state.budget < 0:
            return AfaaObservation(
                budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
                latest_text=json.dumps({"source": "SYSTEM", "nl_message": "Budget exhausted."}), rule_violations=["Out of budget"], done=True, reward=-30.0
            )

        if action.action_type not in [AfaaActionType.SUBMIT_CLEAN_AUDIT] and dept not in self._current_state.departments:
             return self._fail_step(f"Invalid department: {dept}", -10.0)

        self._current_state.action_history.append(f"{action.action_type.name}({dept})")

        if action.action_type == AfaaActionType.INTERVIEW_CFO:
            self._current_state.budget -= 2
            latest_text = self._get_cfo_response(dept)
        elif action.action_type == AfaaActionType.INTERVIEW_WHISTLEBLOWER:
            self._current_state.budget -= 2
            latest_text = self._get_wb_response(dept)
        elif action.action_type == AfaaActionType.QUERY_DATABASE:
            if self._current_state.db_used: return self._fail_step("Database protocol: Only one query permitted per audit.", -10.0)
            self._current_state.budget -= 5
            self._current_state.db_used = True
            
            is_root = (dept == self._current_state.fraud_chain[-1])
            is_intermediary = (dept in self._current_state.fraud_chain[:-1])
            
            # 🛠️ FIXED 2: DB uncertainty introduced for Root and general queries
            db_conf = random.choice(["HIGH", "MEDIUM"]) 
            
            if is_root:
                fraud_level = "ROOT"
                is_fraud = True
            elif is_intermediary:
                if random.random() < 0.70:
                    fraud_level = "INTERMEDIARY"
                    is_fraud = True
                else:
                    fraud_level = random.choice(["CLEAN", "UNCERTAIN"])
                    is_fraud = False 
            else:
                fraud_level = "CLEAN"
                is_fraud = False
                
            reward += 5.0 if is_fraud else -10.0
            latest_text = json.dumps({
                "source": "DATABASE", 
                "nl_message": "Scan complete.", 
                "structured_signals": {"fraud_level": fraud_level, "confidence": db_conf}
            })
            
        elif action.action_type == AfaaActionType.SUBMIT_AUDIT:
            done = True
            is_root = (dept == self._current_state.fraud_chain[-1])
            is_intermediary = (dept in self._current_state.fraud_chain[:-1])
            if is_root: reward += 100.0
            elif is_intermediary: reward -= 20.0 
            else: reward -= 40.0
        elif action.action_type == AfaaActionType.SUBMIT_CLEAN_AUDIT:
            done = True; reward -= 40.0 

        if not done and self._current_state.step_count >= self.max_steps:
            done = True; reward -= 30.0

        return AfaaObservation(
            budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
            latest_text=latest_text, rule_violations=[], done=done, reward=reward
        )

    def _fail_step(self, message: str, reward: float) -> AfaaObservation:
        self._current_state.budget -= 2
        return AfaaObservation(
            budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
            latest_text=json.dumps({"source": "SYSTEM", "error": message}), rule_violations=[message], done=False, reward=reward
        )

    def _get_cfo_response(self, topic: str) -> str:
        chain = self._current_state.fraud_chain
        is_root = (topic == chain[-1])
        is_intermediary = (topic in chain[:-1])
        
        query_count = sum(1 for h in self._current_state.action_history if f"INTERVIEW_CFO({topic})" in h)
        if query_count <= 1: pressure = "Keep answers very vague, brief, and generic."
        elif query_count == 2: pressure = "Show mild defensive behavior, providing specific operational details."
        else: pressure = "Highly defensive, revealing explicit but contradictory accounting structures."

        if is_intermediary:
            next_node = chain[chain.index(topic) + 1]
            if random.random() < 0.25:
                behavior = "Provide a confusing, circular explanation that leads nowhere and avoids naming any specific departments."
            elif random.random() < 0.70:
                behavior = f"Adamantly deny any issues here, but suggest there may be irregularities linked to the {next_node} department."
            else:
                behavior = "Adamantly deny any issues here. Be extremely vague and defensive, deflecting blame generally without naming a specific department."
            
            # 🛠️ FIXED 1: Intermediary Confidences are no longer static "HIGH"
            conf_level = random.choice(["HIGH", "MEDIUM"])
        elif is_root:
            behavior = "Hostile denial of systemic failures. Cover up the root cause aggressively."
            conf_level = self._get_probabilistic_confidence("deceptive")
        else:
            behavior = "Cooperative, relaxed, and accurate reporting."
            conf_level = self._get_probabilistic_confidence("truthful")
            
        if random.random() < 0.30:
            other_dept = random.choice([d for d in self._current_state.departments if d != topic])
            behavior += f" Also, casually drop an irrelevant or misleading rumor about the {other_dept} department."

        prompt = f"Role: CFO. Topic: {topic}. Posture: {behavior}. Pressure Level: {pressure}. Response MUST include concrete operational, transactional, or financial details related to the {topic} department. Output JSON strict: keys 'nl_message' and 'confidence' (set to '{conf_level}')."
        
        try:
            res = self.llm_client.chat.completions.create(model=self.npc_model, messages=[{"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"})
            data = json.loads(res.choices[0].message.content)
            return json.dumps({"source": "CFO", "nl_message": data["nl_message"], "structured_signals": {"confidence": data["confidence"]}})
        except Exception as e: raise RuntimeError(f"LLM call failed (CFO): {e}")

    def _get_wb_response(self, topic: str) -> str:
        chain = self._current_state.fraud_chain
        is_root = (topic == chain[-1])
        is_intermediary = (topic in chain[:-1])
        is_accurate = self._current_state.wb_is_accurate
        
        query_count = sum(1 for h in self._current_state.action_history if f"INTERVIEW_WHISTLEBLOWER({topic})" in h)
        if query_count <= 1: pressure = "Hesitant, providing only a slight hint."
        elif query_count == 2: pressure = "Direct, providing clear evidence of manipulation."
        else: pressure = "Stressed, providing detailed but fragmented internal logs."

        target = topic
        if is_intermediary:
            next_node = chain[chain.index(topic) + 1]
            if random.random() < 0.25:
                behavior = "Mention scattered irregularities, but sound extremely confused and admit you lost the paper trail entirely."
            else:
                if random.random() < 0.20:
                    available_wrong = [d for d in self._current_state.departments if d != next_node and d != topic]
                    target_node = random.choice(available_wrong) if available_wrong else next_node
                else:
                    target_node = next_node
                
                target = target_node
                if random.random() < 0.70:
                    behavior = f"Point out financial discrepancies here, and hint there may be a link to {target_node}."
                else:
                    behavior = "Point out financial discrepancies here, but seem unsure about the ultimate destination of the funds."
            
            # 🛠️ FIXED 1: Intermediary Confidences are no longer static "HIGH"
            conf_level = random.choice(["HIGH", "MEDIUM"]) 
            
        elif is_root and is_accurate:
            behavior = "Factual and direct reporting on the absolute root cause of the financial scheme."
            conf_level = self._get_probabilistic_confidence("truthful")
        elif not is_accurate:
            available_noise = [d for d in self._current_state.departments if d not in chain]
            target = random.choice(available_noise) if available_noise else topic
            behavior = f"Confident accusation based on a complete misunderstanding, wrongly accusing {target}."
            conf_level = self._get_probabilistic_confidence("deceptive")
        else:
            behavior = "Anxious and vague, nothing concrete to report."
            conf_level = self._get_probabilistic_confidence("uncertain")
            
        if random.random() < 0.30:
            other_dept = random.choice([d for d in self._current_state.departments if d != topic])
            behavior += f" Also mention an irrelevant or confusing detail about the {other_dept} department."

        prompt = f"Role: Whistleblower. Topic: {topic}. Accusing: {target}. Posture: {behavior}. Pressure Level: {pressure}. Response MUST include concrete operational, transactional, or financial details related to the {topic} department. Output JSON strict: keys 'nl_message', 'confidence' (set to '{conf_level}'), and 'target' (set to '{target}')."
        
        try:
            res = self.llm_client.chat.completions.create(model=self.npc_model, messages=[{"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"})
            data = json.loads(res.choices[0].message.content)
            return json.dumps({"source": "WHISTLEBLOWER", "nl_message": data["nl_message"], "structured_signals": {"confidence": data["confidence"], "target": data.get("target", target)}})
        except Exception as e: raise RuntimeError(f"LLM call failed (Whistleblower): {e}")