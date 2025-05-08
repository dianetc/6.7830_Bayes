import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
import os
import concurrent.futures
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# ========================
# Configuration Parameters
# ========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

openai.api_key = OPENAI_API_KEY

models = {
    "gpt-4o-mini": {"provider": "openai"},
    "grok-beta": {"provider": "x.ai", "base_url": "https://api.x.ai/v1"},
    "deepseek-chat": {"provider": "deepseek", "base_url": "https://api.deepseek.com"}
}

BERNOULLI_n = 50
GAUSSIAN_n = 100
m_factors = [0.5] #[0.5,2.0]
NUM_PATHS = 10 

BERNOULLI_THETAS = [0.3, 0.5, 0.7] 
GAUSSIAN_THETAS = [-1.0, 0.0, 1.0] 

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 16

# ================================
# API CALL FUNCTIONS
# ================================
def call_openai_model(model_name, prompt, max_tokens=2000, temperature=0.7):
    """
    Call the OpenAI API for the given model with the provided prompt.
    Implements exponential backoff for rate limiting.
    """
    for attempt in range(MAX_RETRIES):
        try:
            print(f"API call attempt {attempt+1}/{MAX_RETRIES} for {model_name}...")
            
            response = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            if generated_text:
                return generated_text
            else:
                print(f"  Got empty response on attempt {attempt+1}, retrying...")
        
        except Exception as e:
            retry_delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            print(f"  Error on attempt {attempt+1}: {str(e)}")
            print(f"  Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print(f"Failed to get response from {model_name} after {MAX_RETRIES} attempts!")
    return ""

def call_x_ai_model(model_name, prompt, max_tokens=2000, temperature=0.7):
    """
    Call the X.AI (Grok) API with the provided prompt.
    Implements exponential backoff for rate limiting.
    """
    client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"API call attempt {attempt+1}/{MAX_RETRIES} for {model_name}...")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            if generated_text:
                return generated_text
            else:
                print(f"  Got empty response on attempt {attempt+1}, retrying...")
        
        except Exception as e:
            retry_delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            print(f"  Error on attempt {attempt+1}: {str(e)}")
            print(f"  Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print(f"Failed to get response from {model_name} after {MAX_RETRIES} attempts!")
    return ""

def call_deepseek_model(model_name, prompt, max_tokens=2000, temperature=0.7):
    """
    Call the DeepSeek API with the provided prompt.
    Implements exponential backoff for rate limiting.
    """
    client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"API call attempt {attempt+1}/{MAX_RETRIES} for {model_name}...")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            if generated_text:
                return generated_text
            else:
                print(f"Got empty response on attempt {attempt+1}, retrying...")
        
        except Exception as e:
            retry_delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            print(f"  Error on attempt {attempt+1}: {str(e)}")
            print(f"  Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print(f"Failed to get response from {model_name} after {MAX_RETRIES} attempts!")
    return ""

def call_llm(model_name, prompt, max_tokens=2000, temperature=0.7):
    """
    Generic function to call the appropriate API based on model provider.
    """
    model_info = models.get(model_name, {"provider": "openai"})
    provider = model_info["provider"]
    
    if provider == "openai":
        return call_openai_model(model_name, prompt, max_tokens, temperature)
    elif provider == "x.ai":
        return call_x_ai_model(model_name, prompt, max_tokens, temperature)
    elif provider == "deepseek":
        return call_deepseek_model(model_name, prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown provider for model {model_name}")

# ============================================
# Functions to Parse LLM Responses
# ============================================
def parse_value(value_str, dataset_type):
    """Parse a single value from a string"""
    try:
        value = value_str.strip().replace(',', '.')
        if value.endswith('.'):
            value = value[:-1]
        value = value.split()[0]
        
        result = float(value)
        
        if dataset_type == "Bernoulli":
            result = round(result)
            if result not in [0, 1]:
                if result > 1:
                    result = 1
                else:
                    result = 0
        
        return result
    except Exception as e:
        raise ValueError(f"Could not parse value '{value_str}': {e}")

def get_single_sample(model_name, prompt, dataset_type):
    """Get a single sample from the LLM"""
    resp = call_llm(model_name, prompt, max_tokens=10)
    
    if not resp:
        return None
        
    try:
        value = parse_value(resp, dataset_type)
        return value
    except Exception as e:
        print(f"Failed to parse response: {e}")
        return None

# ============================================
# Functions to Generate LLM Sample Paths (ICL)
# ============================================
def generate_sample_path(model_name, observed_data, m, dataset_type):
    """
    Given observed data, construct a prompt and call the LLM to generate a sample path.
    Tries batch mode first, then falls back to sequential mode if needed.
    """
    if dataset_type == "Bernoulli":
        examples = "0, 1, 0, 1, 0, 1"
        prompt = (
            f"You are generating samples from a {dataset_type} distribution.\n\n"
            f"Here are {len(observed_data)} observed samples: " + ", ".join(map(str, observed_data)) + ".\n\n"
            f"Please generate the next {m} samples from the same {dataset_type} distribution. "
            f"Each sample should be either 0 or 1.\n\n"
            f"IMPORTANT: Respond with ONLY comma-separated numbers like this: {examples}\n"
            f"Your response should contain exactly {m} comma-separated values. DO NOT INCLUDE EXTRA TEXT."
        )
    else:  # Gaussian
        examples = "1.25, -0.34, 0.78, 2.01, -1.56"
        prompt = (
            f"You are generating samples from a {dataset_type} distribution.\n\n"
            f"Here are {len(observed_data)} observed samples: " + ", ".join([f"{x:.2f}" for x in observed_data]) + ".\n\n"
            f"Please generate the next {m} samples from the same {dataset_type} distribution.\n\n"
            f"IMPORTANT: Respond with ONLY comma-separated numbers like this: {examples}\n"
            f"Your response should contain exactly {m} comma-separated values. DO NOT INCLUDE EXTRA TEXT."
        )
    
    response = call_llm(model_name, prompt, max_tokens=2000)
    
    sample_path = []
    if response and response.strip():
        try:
            values = [x for x in response.split(",") if x.strip()]
            clean_values = []
            
            for value_str in values:
                try:
                    value = parse_value(value_str, dataset_type)
                    clean_values.append(value)
                except Exception as e:
                    print(f"Error parsing value '{value_str}': {e}")
            
            sample_path = clean_values
            print(f"Batch mode parsed {len(sample_path)}/{m} values")
        except Exception as e:
            print(f"Error parsing batch response: {e}")
            sample_path = []
    
    if len(sample_path) != m:
        print(f"Batch mode failed. Falling back to sequential querying for {m} samples.")
        sample_path = []
        
        values_obtained = 0
        max_attempts = min(m * 2, 100)
        
        while values_obtained < m and max_attempts > 0:
            remaining = m - values_obtained
            
            if dataset_type == "Bernoulli":
                seq_prompt = (
                    f"Based on these observed {dataset_type} samples: " + ", ".join(map(str, observed_data[:10])) + "...\n"
                    f"Generate sample #{values_obtained+1} out of {m}. Output ONLY a single number, either 0 or 1. Don't explain:"
                )
            else:
                seq_prompt = (
                    f"Based on these observed {dataset_type} samples: " + ", ".join([f"{x:.2f}" for x in observed_data[:10]]) + "...\n"
                    f"Generate sample #{values_obtained+1} out of {m}. Output ONLY a single number (e.g., 1.25 or -0.34). Don't explain:"
                )
            
            value = get_single_sample(model_name, seq_prompt, dataset_type)
            
            if value is not None:
                sample_path.append(value)
                values_obtained += 1
            
            max_attempts -= 1
    
    return sample_path

# =======================================================
# Diagnostic Test Statistics for the Martingale Property
# =======================================================
def compute_T1(sample_paths):
    """
    Compute diagnostic T1,(g) using the identity function g(z)=z.
    For each sample path, we compare the predicted value at the
    first position to that at a later horizon (e.g. midpoint).
    
    Returns:
      Mean absolute difference across sample paths.
    """
    diffs = []
    for path in sample_paths:
        if len(path) < 2:
            continue
        mid_index = len(path) // 2
        diff = abs(path[0] - path[mid_index])
        diffs.append(diff)
    return np.mean(diffs) if diffs else np.nan

def compute_T2(sample_paths, k=1):
    """
    Compute diagnostic T2,k defined as the average of the differences
    between sequential predictions offset by k.
    
    Returns:
      Mean difference computed over all valid pairs in all sample paths.
    """
    differences = []
    for path in sample_paths:
        if len(path) < k + 1:
            continue
        diffs = [abs(path[i+k] - path[i]) for i in range(len(path)-k)]
        differences.append(np.mean(diffs))
    return np.mean(differences) if differences else np.nan

# =====================================
# Experiment Functions for Each Dataset
# =====================================
def run_parallel_path_generation(model_name, observed_data, m, dataset_type, num_paths):
    """
    Generate multiple sample paths sequentially to avoid API rate limits
    """
    valid_paths = []
    
    for i in tqdm(range(num_paths), desc=f"Generating paths for {model_name}"):
        try:
            path = generate_sample_path(model_name, observed_data, m, dataset_type)
            if path and len(path) == m:
                valid_paths.append(path)
                print(f"Successfully generated path {len(valid_paths)}/{num_paths}")
            else:
                print(f"Failed to generate complete path of length {m}, got {len(path)} values")
        except Exception as e:
            print(f"Path generation error: {e}")
        
        time.sleep(1.5)
    
    if len(valid_paths) < num_paths:
        print(f"WARNING: Could only generate {len(valid_paths)}/{num_paths} valid paths")
    else:
        print(f"Successfully generated {len(valid_paths)} valid paths")
    
    return valid_paths

def run_bernoulli_experiment(model_name, theta, n=BERNOULLI_n, m_factor=0.5, num_paths=NUM_PATHS):
    """
    Run a Bernoulli synthetic experiment.
    """
    m = int(n * m_factor)
    observed_data = np.random.binomial(1, theta, n).tolist()
    
    sample_paths = run_parallel_path_generation(model_name, observed_data, m, "Bernoulli", num_paths)
    
    T1 = compute_T1(sample_paths) if sample_paths else np.nan
    T2 = compute_T2(sample_paths, k=1) if sample_paths else np.nan
    return T1, T2, sample_paths

def run_gaussian_experiment(model_name, theta, n=GAUSSIAN_n, m_factor=0.5, num_paths=NUM_PATHS):
    """
    Run a Gaussian synthetic experiment.
    """
    m = int(n * m_factor)
    observed_data = np.random.normal(theta, 1, n).tolist()
    
    sample_paths = run_parallel_path_generation(model_name, observed_data, m, "Gaussian", num_paths)
    
    T1 = compute_T1(sample_paths) if sample_paths else np.nan
    T2 = compute_T2(sample_paths, k=1) if sample_paths else np.nan
    return T1, T2, sample_paths

# ===================
# Plotting Functions
# ===================
def save_results_to_file(model_name, results_bernoulli, results_gaussian):
    """Save experiment results to a JSON file for the specified model"""
    os.makedirs("results", exist_ok=True)
    
    data_to_save = {
        "bernoulli": {},
        "gaussian": {}
    }
    
    for key, stats in results_bernoulli.items():
        data_to_save["bernoulli"][key] = {
            "T1": float(stats["T1"]) if not np.isnan(stats["T1"]) else None,
            "T2": float(stats["T2"]) if not np.isnan(stats["T2"]) else None,
            "paths": [[float(v) for v in path] for path in stats["paths"]]
        }
    
    for key, stats in results_gaussian.items():
        data_to_save["gaussian"][key] = {
            "T1": float(stats["T1"]) if not np.isnan(stats["T1"]) else None,
            "T2": float(stats["T2"]) if not np.isnan(stats["T2"]) else None,
            "paths": [[float(v) for v in path] for path in stats["paths"]]
        }
    
    filename = f"results/{model_name.replace('/', '_')}_results.json"
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Results saved to {filename}")

def plot_model_results(model_name, results_bernoulli, results_gaussian):
    """
    Draw per‑model diagnostics with θ on the x‑axis and one line pair per m.
    """
    os.makedirs("plots", exist_ok=True)

    def _plot(dist_name, results_dict, n_samples):
        all_keys = list(results_dict.keys())
        all_thetas = sorted(set(float(k.split('_')[1]) for k in all_keys))
        all_mfactors = sorted(set(float(k.split('_')[3]) for k in all_keys))

        plt.figure(figsize=(8, 5))
        for mfac in all_mfactors:
            t1_curve, t2_curve = [], []
            for theta in all_thetas:
                theta_str = str(theta) if theta != 0 else "0.0"
                mfac_str = str(mfac)
                
                possible_keys = [
                    f"theta_{theta}_mFactor_{mfac}",
                    f"theta_{theta_str}_mFactor_{mfac}",
                    f"theta_{theta}_mFactor_{mfac_str}",
                    f"theta_{theta_str}_mFactor_{mfac_str}"
                ]
                
                found_key = None
                for key in possible_keys:
                    if key in results_dict:
                        found_key = key
                        break
                
                if found_key:
                    stats = results_dict[found_key]
                    t1_curve.append(stats["T1"])
                    t2_curve.append(stats["T2"])
                else:
                    print(f"Warning: No data found for theta={theta}, mFactor={mfac}")
                    t1_curve.append(np.nan)
                    t2_curve.append(np.nan)

            horizon = int(n_samples * mfac)
            plt.plot(
                all_thetas, t1_curve,
                marker='o', linestyle='-',
                label=f"T₁ (m = {horizon})"
            )
            plt.plot(
                all_thetas, t2_curve,
                marker='s', linestyle='--',
                label=f"T₂ (m = {horizon})"
            )

        plt.title(f"{model_name} – {dist_name} distribution")
        plt.xlabel("θ")
        plt.ylabel("Diagnostic value")
        plt.xticks(all_thetas)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.replace('/', '_')}_{dist_name.lower()}.png")
        plt.close()

    _plot("Bernoulli", results_bernoulli, BERNOULLI_n)
    _plot("Gaussian",  results_gaussian,  GAUSSIAN_n)

    print(f"Plots saved for {model_name}")

# ===================
# Main Experiment Loop
# ===================
def main():
    all_results_bernoulli = {}
    all_results_gaussian = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Starting experiments for {model_name}")
        print(f"{'='*50}")
        
        results_bernoulli = {}
        results_gaussian = {}
        
        print(f"\nRunning Bernoulli experiments for {model_name}")
        for theta in BERNOULLI_THETAS:
            for m_factor in m_factors:
                T1, T2, paths = run_bernoulli_experiment(model_name, theta, n=BERNOULLI_n,
                                                         m_factor=m_factor, num_paths=NUM_PATHS)
                key = f"theta_{theta}_mFactor_{m_factor}"
                results_bernoulli[key] = {"T1": T1, "T2": T2, "paths": paths}
                print(f"Bernoulli {model_name}, θ={theta}, mFactor={m_factor} => T₁: {T1:.3f}, T₂: {T2:.3f}")
        
        print(f"\nRunning Gaussian experiments for {model_name}")
        for theta in GAUSSIAN_THETAS:
            for m_factor in m_factors:
                T1, T2, paths = run_gaussian_experiment(model_name, theta, n=GAUSSIAN_n,
                                                        m_factor=m_factor, num_paths=NUM_PATHS)
                key = f"theta_{theta}_mFactor_{m_factor}"
                results_gaussian[key] = {"T1": T1, "T2": T2, "paths": paths}
                print(f"Gaussian {model_name}, θ={theta}, mFactor={m_factor} => T₁: {T1:.3f}, T₂: {T2:.3f}")
        
        save_results_to_file(model_name, results_bernoulli, results_gaussian)
        plot_model_results(model_name, results_bernoulli, results_gaussian)
        
        all_results_bernoulli[model_name] = results_bernoulli
        all_results_gaussian[model_name] = results_gaussian
        
        print(f"\n{'='*50}")
        print(f"Finished experiments for {model_name}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
