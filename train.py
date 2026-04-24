import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions
from client import AfaaEnvClient
import asyncio
from models import AfaaAction

env = AfaaEnvClient(base_url="http://localhost:8000") # Your running HF Space or Docker

def extract_json_action(text: str) -> AfaaAction:
    # Helper to parse the LLM's raw text into your Pydantic schema
    try:
        # Simple extraction logic (assumes LLM outputs raw JSON block)
        start = text.find('{')
        end = text.rfind('}') + 1
        return AfaaAction.model_validate_json(text[start:end])
    except:
        # Fallback invalid action
        return AfaaAction(thought="Failed parsing", action_type="SUBMIT_CLEAN_AUDIT")

def rollout_once(trainer, env, tokenizer, dataset_prompt, system_prompt, max_turns=10):
    """Executes one full episode."""
    result = env.reset()
    observation = result.observation
    
    prompt_ids, completion_ids, logprobs, rewards = [], [], [], []
    history = []

    for _ in range(max_turns):
        if result.done:
            break
            
        # Build prompt using current observation and history
        user_prompt = f"{dataset_prompt}\nObservation: {observation.latest_text}\nBudget: {observation.budget_remaining}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Generate with vLLM
        outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        
        prompt_ids.extend(outputs["prompt_ids"])
        completion_ids.extend(outputs["completion_ids"])
        logprobs.extend(outputs["logprobs"])
        
        completion_text = tokenizer.decode(outputs["completion_ids"], skip_special_tokens=True)
        action = extract_json_action(completion_text)
        
        # Step Environment
        result = env.step(action)
        observation = result.observation
        rewards.append(float(result.reward))
        
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "reward": rewards[-1] if rewards else 0.0 # Terminal reward emphasis
    }

# --- FIXED: Make the single episode runner async ---
async def rollout_once_async(trainer, env, tokenizer, dataset_prompt, system_prompt, max_turns=10):
    result = await env.reset() # Await properly
    observation = result.observation
    
    prompt_ids, completion_ids, logprobs, rewards = [], [], [], []

    for _ in range(max_turns):
        if result.done:
            break
            
        user_prompt = f"{dataset_prompt}\nObservation: {observation.latest_text}\nBudget: {observation.budget_remaining}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        
        prompt_ids.extend(outputs["prompt_ids"])
        completion_ids.extend(outputs["completion_ids"])
        logprobs.extend(outputs["logprobs"])
        
        completion_text = tokenizer.decode(outputs["completion_ids"], skip_special_tokens=True)
        action = extract_json_action(completion_text)
        
        result = await env.step(action) # Await properly
        observation = result.observation
        rewards.append(float(result.reward))
        
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "reward": rewards[-1] if rewards else 0.0 
    }

def rollout_func(prompts, trainer=None):
    batch_results = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "reward": []}
    system_prompt = "You are a Sovereign Auditor. Output strict JSON with 'thought', 'action_type', and 'department'."
    
    # --- FIXED: Synchronous wrapper for TRL to execute the async rollouts ---
    async def run_batch():
        episodes = []
        for prompt in prompts:
            ep = await rollout_once_async(trainer, env, trainer.processing_class, prompt, system_prompt)
            episodes.append(ep)
        return episodes

    episodes = asyncio.run(run_batch())
    
    for ep in episodes:
        for k in batch_results:
            batch_results[k].append(ep[k])
            
    return batch_results

# Define Reward Function expected by TRL
def reward_total(completions, **kwargs):
    return kwargs.get("reward", [0.0] * len(completions))

# Execute Training
config = GRPOConfig(output_dir="afaa-grpo-agent", max_completion_length=150, use_vllm=True)
dataset = Dataset.from_dict({"prompt": ["Find the corporate fraud."] * 100})

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B", # Or Meta Llama
    reward_funcs=[reward_total],
    train_dataset=dataset,
    args=config,
    rollout_func=rollout_func
)

trainer.train()