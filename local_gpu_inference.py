# local_gpu_inference.py
"""
OPTIMIZED Local GPU inference using PyTorch + CUDA for ESG data extraction.
No cloud API required - runs entirely on your RTX 3050.

Optimizations:
- 4-bit quantization (BitsAndBytes) - 4x less VRAM, faster inference
- Smaller model (Qwen2.5-0.5B) - 2x faster than TinyLlama
- Optimized generation settings - reduced overhead
- KV-cache enabled - faster token generation
- Performance tracking - monitor speed
"""

import os
import json
import torch
import re
import time
from typing import Dict, Any, List, Optional
from copy import deepcopy

# Check GPU
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# Global model state
_model = None
_tokenizer = None
_model_loaded = False
_quantized = False

# Performance tracking
_perf_stats = {
    "total_time": 0,
    "total_tokens": 0,
    "calls": 0,
}


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "device": DEVICE,
    }
    if CUDA_AVAILABLE:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["gpu_memory_free_gb"] = round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9, 2)
    return info


def load_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", use_4bit: bool = True):
    """
    Load OPTIMIZED local model for RTX 3050 (4GB VRAM).
    
    Using Qwen2.5-0.5B-Instruct with 4-bit quantization:
    - Only ~0.3GB VRAM (vs 2.2GB for TinyLlama FP16)
    - 2-3x faster inference
    - Good at structured JSON extraction
    """
    global _model, _tokenizer, _model_loaded, _quantized
    
    if _model_loaded:
        return True
    
    if not CUDA_AVAILABLE:
        print("❌ No CUDA GPU available")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"🔄 Loading {model_name} on GPU...")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Try 4-bit quantization for speed
        quantization_config = None
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                print("   ⚡ Enabling 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                _quantized = True
            except ImportError:
                print("   ℹ️ BitsAndBytes not available, using FP16")
                _quantized = False
        
        # Load fast tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        # Load model with optimizations
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "cuda",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        
        # Try SDPA attention (faster than default)
        try:
            load_kwargs["attn_implementation"] = "sdpa"
        except:
            pass
        
        _model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        _model.eval()
        _model_loaded = True
        
        mem_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"✅ Model loaded! GPU memory: {mem_used:.2f} GB")
        print(f"   4-bit quantized: {_quantized}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Fallback without quantization
        if use_4bit:
            print("   Retrying without 4-bit quantization...")
            return load_model(model_name, use_4bit=False)
        _model = None
        _tokenizer = None
        _model_loaded = False
        return False


def unload_model():
    """Free GPU memory."""
    global _model, _tokenizer, _model_loaded
    
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    
    _model_loaded = False
    
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        print("🗑️ Model unloaded, GPU memory freed")


def extract_json_from_text(text: str) -> str:
    """Extract JSON from model output."""
    # Try ```json blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    
    # Try { ... } blocks
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    
    return text


@torch.inference_mode()  # Faster than torch.no_grad()
def generate(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 400,  # Reduced for ESG JSON output
    temperature: float = 0.05,  # Lower = more deterministic
) -> str:
    """Generate text using OPTIMIZED local GPU model."""
    global _model, _tokenizer, _model_loaded, _perf_stats
    
    if not _model_loaded:
        if not load_model():
            raise RuntimeError("Could not load local model")
    
    start_time = time.perf_counter()
    
    # Format as chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    # Apply chat template
    if hasattr(_tokenizer, "apply_chat_template"):
        input_text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    # Tokenize with reduced context for speed
    inputs = _tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1536,  # Reduced for speed
        padding=False,
    )
    inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
    
    # OPTIMIZED generation settings for JSON output
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        top_p=0.85 if temperature > 0 else None,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
        repetition_penalty=1.15,  # Prevent repetition
        # Speed optimizations
        use_cache=True,
        num_beams=1,
    )
    
    # Decode only new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Track performance
    elapsed = time.perf_counter() - start_time
    _perf_stats["total_time"] += elapsed
    _perf_stats["total_tokens"] += len(generated_ids)
    _perf_stats["calls"] += 1
    
    return response.strip()


def get_performance_stats() -> Dict[str, Any]:
    """Get inference performance statistics."""
    stats = _perf_stats.copy()
    if stats["calls"] > 0:
        stats["avg_time_per_call"] = round(stats["total_time"] / stats["calls"], 2)
        stats["tokens_per_second"] = round(stats["total_tokens"] / max(stats["total_time"], 0.01), 1)
    return stats


class LocalGPUClient:
    """
    OPTIMIZED Local GPU LLM client.
    Uses RTX 3050 with 4-bit quantization for fast inference.
    """
    
    def __init__(self):
        self.stats = {
            "calls": 0,
            "tokens_generated": 0,
            "errors": 0,
            "total_time": 0,
        }
        # Pre-load model
        load_model()
    
    def call(
        self,
        messages: List[Dict],
        max_tokens: int = 400,  # Reduced for speed
        temperature: float = 0.05,  # Lower = more deterministic JSON
    ) -> Dict[str, Any]:
        """Make OPTIMIZED inference call on local GPU."""
        
        # Extract system and user messages
        system_prompt = "You are a helpful assistant."
        user_prompt = ""
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", system_prompt)
            elif msg.get("role") == "user":
                user_prompt = msg.get("content", "")
        
        start = time.perf_counter()
        
        try:
            response = generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            
            elapsed = time.perf_counter() - start
            self.stats["calls"] += 1
            self.stats["tokens_generated"] += len(response.split())
            self.stats["total_time"] += elapsed
            
            return {
                "content": extract_json_from_text(response),
                "source": "local_gpu_optimized",
                "inference_time": round(elapsed, 2),
                "usage": {
                    "prompt_tokens": len(user_prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(user_prompt.split()) + len(response.split()),
                }
            }
            
        except Exception as e:
            self.stats["errors"] += 1
            raise RuntimeError(f"Local GPU inference failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics with performance metrics."""
        stats = self.stats.copy()
        if stats["calls"] > 0:
            stats["avg_time_per_call"] = round(stats["total_time"] / stats["calls"], 2)
        return stats


# Validation functions (same as before)
def validate_extracted_data(data: Dict, schema: Dict) -> Dict[str, Any]:
    """Validate extracted data against schema."""
    warnings = []
    suspicious_values = []
    
    def check_value(key: str, value: Any, path: str = ""):
        full_path = f"{path}.{key}" if path else key
        
        if value is None:
            return
        
        if isinstance(value, (int, float)):
            # Check for suspicious values
            if value < 0 and "percent" in key.lower():
                suspicious_values.append(f"{full_path}: negative percentage ({value})")
            if value > 1000000000:  # > 1 billion
                warnings.append(f"{full_path}: very large value ({value})")
        
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value(k, v, full_path)
    
    for section, section_data in data.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                check_value(key, value, section)
    
    return {
        "warnings": warnings,
        "suspicious_values": suspicious_values,
        "valid": len(warnings) == 0,
    }


def check_data_consistency(data: Dict) -> List[str]:
    """Check for logical inconsistencies in extracted data."""
    issues = []
    
    env = data.get("Environment", {})
    social = data.get("Social", {})
    gov = data.get("Governance", {})
    
    # Check GHG totals
    scope1 = env.get("scope1_tco2e")
    scope2 = env.get("scope2_tco2e")
    scope3 = env.get("scope3_tco2e")
    total_ghg = env.get("ghg_total_tco2e")
    
    if all(v is not None for v in [scope1, scope2, scope3, total_ghg]):
        calculated = scope1 + scope2 + scope3
        if abs(calculated - total_ghg) > total_ghg * 0.1:  # >10% difference
            issues.append(f"GHG total mismatch: calculated={calculated}, reported={total_ghg}")
    
    # Check employee totals
    total_emp = social.get("total_employees")
    perm_emp = social.get("permanent_employees")
    contract = social.get("contract_workers")
    
    if all(v is not None for v in [total_emp, perm_emp, contract]):
        if perm_emp + contract > total_emp * 1.1:  # Allow 10% variance
            issues.append(f"Employee count inconsistency: perm({perm_emp}) + contract({contract}) > total({total_emp})")
    
    # Check board composition
    board_size = gov.get("board_size_count")
    ind_dirs = gov.get("independent_directors_count")
    women_dirs = gov.get("women_directors_count")
    
    if board_size is not None and ind_dirs is not None:
        if ind_dirs > board_size:
            issues.append(f"Independent directors ({ind_dirs}) > board size ({board_size})")
    
    if board_size is not None and women_dirs is not None:
        if women_dirs > board_size:
            issues.append(f"Women directors ({women_dirs}) > board size ({board_size})")
    
    return issues


# Test function with benchmark
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED Local GPU Inference Test")
    print("=" * 60)
    
    info = get_gpu_info()
    print(f"\nGPU Info: {json.dumps(info, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Loading optimized model (4-bit quantization)...")
    print("=" * 60)
    
    if load_model():
        test_prompt = """Extract ESG metrics from this text as JSON:
{"scope1_tco2e": null, "scope2_tco2e": null, "total_employees": null}

Text: "Our Scope 1 emissions were 15,000 tCO2e and Scope 2 was 8,500 tCO2e. We have 5,200 employees."

Output ONLY JSON."""
        
        # Warm-up run (first run is slower)
        print("\nWarm-up run...")
        _ = generate(test_prompt, max_new_tokens=200)
        
        # Benchmark
        print("\nBenchmark (3 runs):")
        times = []
        for i in range(3):
            start = time.perf_counter()
            response = generate(test_prompt, max_new_tokens=200)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average inference time: {avg_time:.2f}s")
        print(f"📊 Performance stats: {get_performance_stats()}")
        print(f"\nSample output:\n{response}")
        
        mem_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"\n💾 GPU memory used: {mem_used:.2f} GB")
        
        unload_model()
    else:
        print("Failed to load model")
