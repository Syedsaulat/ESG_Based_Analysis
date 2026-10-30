# hybrid_inference.py
"""
Hybrid inference: Groq cloud (primary) with local GPU fallback on rate limits.
Includes error checking and data validation.
"""

import os
import json
import time
import torch
from typing import Optional, Dict, Any, List
from copy import deepcopy

# Check GPU availability
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# Global state
_local_model = None
_local_tokenizer = None
_use_local_fallback = False
_rate_limit_until = 0  # timestamp when rate limit expires


def get_device_info() -> Dict[str, Any]:
    """Return GPU info for status display."""
    info = {"cuda_available": CUDA_AVAILABLE, "device": DEVICE}
    if CUDA_AVAILABLE:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def load_local_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Load a small local model for GPU fallback.
    Qwen2.5-1.5B fits in ~3GB VRAM - good for RTX 3050.
    """
    global _local_model, _local_tokenizer
    
    if _local_model is not None:
        return  # Already loaded
    
    if not CUDA_AVAILABLE:
        print("⚠️ No GPU available for local fallback")
        return
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"🔄 Loading local fallback model: {model_name}")
        print(f"   Device: {DEVICE}")
        
        # Determine dtype based on GPU
        if torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        mem_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"✅ Local model loaded (GPU memory: {mem_used:.2f} GB)")
        
    except Exception as e:
        print(f"❌ Failed to load local model: {e}")
        _local_model = None
        _local_tokenizer = None


def unload_local_model():
    """Free GPU memory by unloading local model."""
    global _local_model, _local_tokenizer
    
    if _local_model is not None:
        del _local_model
        _local_model = None
    if _local_tokenizer is not None:
        del _local_tokenizer
        _local_tokenizer = None
    
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        print("🗑️ Local model unloaded")


def local_generate(prompt: str, system_prompt: str, max_tokens: int = 1500) -> str:
    """Generate using local GPU model."""
    global _local_model, _local_tokenizer
    
    if _local_model is None:
        load_local_model()
    
    if _local_model is None:
        raise RuntimeError("Local model not available")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    # Apply chat template
    input_text = _local_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = _local_tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=4096
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = _local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=_local_tokenizer.pad_token_id,
                eos_token_id=_local_tokenizer.eos_token_id,
            )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = _local_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return extract_json_from_text(response)


def extract_json_from_text(text: str) -> str:
    """Extract JSON from model response."""
    import re
    
    # Try ```json blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    
    # Try { ... } blocks
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    
    return text


class HybridLLMClient:
    """
    Hybrid LLM client: uses Groq primarily, falls back to local GPU on rate limits.
    """
    
    def __init__(self, groq_model: str, groq_api_key: Optional[str] = None):
        from groq import Groq
        
        self.groq_model = groq_model
        self.groq_client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
        self.rate_limit_until = 0
        self.consecutive_errors = 0
        self.max_retries = 3
        
        # Stats
        self.stats = {
            "groq_calls": 0,
            "local_calls": 0,
            "errors": 0,
            "rate_limits": 0,
        }
    
    def call(
        self,
        messages: List[Dict],
        max_tokens: int = 2500,
        temperature: float = 0,
    ) -> Dict[str, Any]:
        """
        Make LLM call with automatic fallback.
        Returns dict with 'content' and 'source' keys.
        """
        current_time = time.time()
        
        # Check if we should use local due to rate limit
        if current_time < self.rate_limit_until and CUDA_AVAILABLE:
            return self._call_local(messages, max_tokens)
        
        # Try Groq first
        try:
            return self._call_groq(messages, max_tokens, temperature)
        except Exception as e:
            error_str = str(e).lower()
            
            # Rate limit - switch to local
            if "rate_limit" in error_str or "rate limit" in error_str:
                self.stats["rate_limits"] += 1
                self.rate_limit_until = current_time + 60  # 60 second cooldown
                print(f"⚠️ Groq rate limit hit - switching to local GPU for 60s")
                
                if CUDA_AVAILABLE:
                    return self._call_local(messages, max_tokens)
                else:
                    print("   No GPU available, waiting 60s...")
                    time.sleep(60)
                    return self._call_groq(messages, max_tokens, temperature)
            
            # Other errors - retry with backoff
            self.stats["errors"] += 1
            self.consecutive_errors += 1
            
            if self.consecutive_errors <= self.max_retries:
                wait_time = 2 ** self.consecutive_errors
                print(f"⚠️ Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self.call(messages, max_tokens, temperature)
            
            raise e
    
    def _call_groq(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Call Groq API."""
        resp = self.groq_client.chat.completions.create(
            model=self.groq_model,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self.stats["groq_calls"] += 1
        self.consecutive_errors = 0
        
        return {
            "content": resp.choices[0].message.content,
            "source": "groq",
            "usage": {
                "prompt": resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
                "total": resp.usage.total_tokens,
            } if hasattr(resp, "usage") else None,
        }
    
    def _call_local(
        self,
        messages: List[Dict],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Call local GPU model."""
        system_prompt = ""
        user_prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]
        
        content = local_generate(user_prompt, system_prompt, max_tokens)
        
        self.stats["local_calls"] += 1
        self.consecutive_errors = 0
        
        return {
            "content": content,
            "source": "local_gpu",
            "usage": None,
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Return usage statistics."""
        return self.stats.copy()


# ------------------------------
# Data Validation & Error Checking
# ------------------------------

def validate_extracted_data(data: Dict, schema: Dict) -> Dict[str, Any]:
    """
    Validate extracted data against schema and check for discrepancies.
    Returns validation report.
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "coverage": 0,
        "null_fields": [],
        "suspicious_values": [],
    }
    
    def check_field(path: str, value: Any, expected_type: Any):
        # Check for null
        if value is None:
            report["null_fields"].append(path)
            return
        
        # Check numeric bounds
        if isinstance(value, (int, float)):
            # Percentage fields should be 0-100
            if "percent" in path.lower():
                if value < 0 or value > 100:
                    report["warnings"].append(f"{path}: {value} outside 0-100 range")
                    report["suspicious_values"].append((path, value, "percentage out of range"))
            
            # Count fields should be non-negative
            if "count" in path.lower() and value < 0:
                report["warnings"].append(f"{path}: negative count ({value})")
                report["suspicious_values"].append((path, value, "negative count"))
            
            # Very large numbers might be errors
            if abs(value) > 1e12:
                report["warnings"].append(f"{path}: unusually large value ({value})")
                report["suspicious_values"].append((path, value, "unusually large"))
    
    def traverse(data: Dict, schema: Dict, prefix: str = ""):
        for key, expected in schema.items():
            path = f"{prefix}.{key}" if prefix else key
            value = data.get(key)
            
            if isinstance(expected, dict) and not isinstance(expected, type(None)):
                if isinstance(value, dict):
                    traverse(value, expected, path)
                elif value is not None:
                    report["errors"].append(f"{path}: expected dict, got {type(value).__name__}")
                    report["valid"] = False
            else:
                check_field(path, value, expected)
    
    traverse(data, schema)
    
    # Calculate coverage
    total_fields = json.dumps(schema).count("null")
    filled_fields = total_fields - len(report["null_fields"])
    report["coverage"] = round(filled_fields / total_fields * 100, 1) if total_fields > 0 else 0
    
    return report


def check_data_consistency(data: Dict) -> List[str]:
    """
    Check for logical inconsistencies in the extracted data.
    """
    issues = []
    
    env = data.get("Environment", {})
    soc = data.get("Social", {})
    gov = data.get("Governance", {})
    
    # Environment checks
    scope1 = env.get("scope1_tco2e")
    scope2 = env.get("scope2_tco2e")
    scope3 = env.get("scope3_tco2e")
    ghg_total = env.get("ghg_total_tco2e")
    
    if all(v is not None for v in [scope1, scope2, scope3, ghg_total]):
        expected_total = scope1 + scope2 + scope3
        if abs(ghg_total - expected_total) > expected_total * 0.1:  # 10% tolerance
            issues.append(f"GHG total ({ghg_total}) doesn't match sum of scopes ({expected_total})")
    
    renewable_pct = env.get("renewable_energy_percent")
    if renewable_pct is not None and (renewable_pct < 0 or renewable_pct > 100):
        issues.append(f"Renewable energy percent ({renewable_pct}) outside valid range")
    
    # Social checks
    male_pct = soc.get("male_percent")
    female_pct = soc.get("female_percent")
    if male_pct is not None and female_pct is not None:
        if abs((male_pct + female_pct) - 100) > 5:  # 5% tolerance for rounding
            issues.append(f"Male ({male_pct}%) + Female ({female_pct}%) doesn't sum to ~100%")
    
    total_emp = soc.get("total_employees")
    perm_emp = soc.get("permanent_employees")
    contract_emp = soc.get("contract_workers")
    if all(v is not None for v in [total_emp, perm_emp, contract_emp]):
        if perm_emp + contract_emp > total_emp * 1.1:  # 10% tolerance
            issues.append(f"Permanent + contract workers exceeds total employees")
    
    # Governance checks
    board_size = gov.get("board_size_count")
    ind_directors = gov.get("independent_directors_count")
    if board_size is not None and ind_directors is not None:
        if ind_directors > board_size:
            issues.append(f"Independent directors ({ind_directors}) exceeds board size ({board_size})")
    
    return issues


# ------------------------------
# Quick test
# ------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Hybrid Inference Module")
    print("=" * 50)
    
    info = get_device_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    if CUDA_AVAILABLE:
        print("\n✅ GPU fallback available")
        print("   Will switch to local model on Groq rate limits")
    else:
        print("\n⚠️ No GPU - will wait on rate limits")
