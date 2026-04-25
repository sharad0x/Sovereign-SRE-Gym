import json
import threading
import random
from pathlib import Path

class GlobalMemory:
    """
    Thread-safe Deep Adversarial Memory System.
    Profiles agent behavior to generate targeted exploits and scale difficulty.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalMemory, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.filepath = Path(__file__).parent / "global_memory.json"
        self.history_window = 10
        self.stats = {
            "episodes_played": 0,
            "recent_wins": [],
            "recent_steps": [],
            "difficulty_level": 1,
            "db_dependency_score": 0.0,
            "shallow_exploration_count": 0,
            "failed_nodes": []
        }
        self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    self.stats = json.load(f)
            except Exception as e:
                print(f"⚠️ [AFAA Backend Log] Failed to load memory: {e}")

    def _save(self):
        try:
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, "w") as f:
                json.dump(self.stats, f)
            temp_file.replace(self.filepath)
        except Exception as e:
            print(f"⚠️ [AFAA Backend Log] Failed to save memory: {e}")

    def record_episode(self, won: bool, db_used: bool, steps: int, target_dept: str = None):
        """Records outcome and performs deep adversarial profiling."""
        with self._lock:
            self.stats["episodes_played"] += 1
            self.stats["recent_wins"].append(1 if won else 0)
            self.stats["recent_steps"].append(steps)
            
            if not won and target_dept:
                self.stats["failed_nodes"].append(target_dept)
                # Keep only recent failures
                if len(self.stats["failed_nodes"]) > 5:
                    self.stats["failed_nodes"].pop(0)
                
                # Detect shallow guessing
                if steps < 5:
                    self.stats["shallow_exploration_count"] += 1
            elif won:
                self.stats["shallow_exploration_count"] = max(0, self.stats["shallow_exploration_count"] - 1)

            if len(self.stats["recent_wins"]) > self.history_window:
                self.stats["recent_wins"].pop(0)
                self.stats["recent_steps"].pop(0)

            # Profile Database Dependency
            if db_used:
                self.stats["db_dependency_score"] = min(1.0, self.stats.get("db_dependency_score", 0.0) + 0.1)
            else:
                self.stats["db_dependency_score"] = max(0.0, self.stats.get("db_dependency_score", 0.0) - 0.1)

            # Curriculum Scaling
            if len(self.stats["recent_wins"]) >= 5:
                win_rate = sum(self.stats["recent_wins"]) / len(self.stats["recent_wins"])
                current_level = self.stats.get("difficulty_level", 1)
                
                if win_rate >= 0.70 and current_level < 5:
                    self.stats["difficulty_level"] = current_level + 1
                    self.stats["recent_wins"] = []; self.stats["recent_steps"] = []
                elif win_rate <= 0.30 and current_level > 1:
                    self.stats["difficulty_level"] = current_level - 1
                    self.stats["recent_wins"] = []; self.stats["recent_steps"] = []

            self._save()

    def get_difficulty_config(self) -> dict:
        """Outputs dynamic parameters heavily skewed by the agent's weaknesses."""
        with self._lock:
            level = self.stats.get("difficulty_level", 1)
            db_dep = self.stats.get("db_dependency_score", 0.0)
            shallow_count = self.stats.get("shallow_exploration_count", 0)
            
            config = {
                "level": level,
                "num_depts": min(7 + (level - 1), 12),
                "num_intermediaries": min(3 + (level // 2) + (1 if shallow_count > 3 else 0), 6), # Force deeper chains if guessing early
                "mutation_prob": min(0.15 + (level * 0.05), 0.40),
                "noise_prob": min(0.20 + (level * 0.05), 0.45),
                "dead_end_prob": min(0.15 + (level * 0.05), 0.40),
                "clue_prob": max(0.80 - (level * 0.05), 0.50),
                "wrong_target_prob": min(0.10 + (level * 0.05), 0.35),
                "starting_mode": "INDEPENDENT",
                "wb_inaccuracy_prob": min(0.20 + (level * 0.05), 0.50),
                "cfo_evasive_prob": min(0.15 + (level * 0.05), 0.40),
                # Adversarial Exploits
                "failed_nodes": self.stats.get("failed_nodes", [])[-3:],
                "db_hallucination_rate": min(0.40, 0.05 + (db_dep * 0.35)) # DB lies if relied upon too heavily
            }
            return config