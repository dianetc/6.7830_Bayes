import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
import requests
import os
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv

# ========================
# Configuration Parameters
# ========================
# Load API key from .env file
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

models = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen-2-Instruct-72B": "Qwen/Qwen2-72B-Instruct",
    "Maestro-Reasoning": "arcee-ai/maestro-reasoning",
    "Meta-Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

BERNOULLI_n = 50
GAUSSIAN_n =  100  #50 #100
m_factors = [0.5, 2.0]
NUM_PATHS =  10  #25  # Reduced to complete faster

MAX_WORKERS = 3 
API_RATE_LIMIT = 4

# ================================
# API CALL USING TOGETHER API
# ================================
def call_llm(model_name, prompt, max_tokens=2000, temperature=0.7, max_retries=3):
    """
    Call the LLM API for the given model with the provided prompt using the Together API.
    Will retry up to max_retries times if the API call fails or returns an empty response.
    Includes rate limiting to avoid API throttling.
    
    Parameters:
      model_name: The model identifier in our models dictionary
      prompt: The text prompt to send to the model
      max_tokens: Maximum number of tokens to generate
      temperature: Controls randomness (lower = more deterministic)
      max_retries: Maximum number of retries for failed API calls
      
    Returns:
      Generated text from the model or empty string if all retries fail
    """

    if not hasattr(call_llm, 'last_call_time'):
        call_llm.last_call_time = 0
        

    current_time = time.time()
    time_since_last_call = current_time - call_llm.last_call_time
    if time_since_last_call < 1.0/API_RATE_LIMIT:
        sleep_time = (1.0/API_RATE_LIMIT) - time_since_last_call
        time.sleep(sleep_time)
    
    call_llm.last_call_time = time.time()
    for attempt in range(max_retries):
        try:
            print(f"API call attempt {attempt+1}/{max_retries} for {model_name}...")
            
            model_id = models[model_name]
            
            url = "https://api.together.xyz/v1/completions"
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["\n\n"]  
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("text", "").strip()
                
                if generated_text:
                    return generated_text
                else:
                    print(f"  Got empty response on attempt {attempt+1}, retrying...")
                    time.sleep(1)
            else:
                print(f"  API Error ({response.status_code}) on attempt {attempt+1}: {response.text}")
                time.sleep(2)
                
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {str(e)}")
            time.sleep(2)
    
    print(f"Failed to get response from {model_name} after {max_retries} attempts!")
    return ""

# ============================================
# Functions to Generate LLM Sample Paths (ICL)
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

def get_single_sample(model_name, prompt, dataset_type, max_retries=3):
    """Get a single sample from the LLM via API with retries"""
    for attempt in range(max_retries):
        resp = call_llm(model_name, prompt, max_tokens=10, max_retries=2)
        
        if not resp:
            continue
            
        try:
            value = parse_value(resp, dataset_type)
            return value
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed to parse: {e}")
            time.sleep(0.5)
    
    return None

def generate_sample_path(model_name, observed_data, m, dataset_type):
    """
    Given observed data (a list of numbers), construct a prompt and call the LLM to generate a sample path.
    Attempts batch mode first, then falls back to parallel sequential mode.
    
    Parameters:
      model_name: string identifier for the LLM.
      observed_data: list of numeric observed values.
      m: number of predictions (sample path length).
      dataset_type: "Bernoulli" or "Gaussian" (used in the prompt).
    
    Returns:
      sample_path: list of floats (the predicted future samples from the LLM).
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
    
    response = call_llm(model_name, prompt, max_tokens=2000, max_retries=3)
    
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
        print(f"Batch mode failed. Falling back to parallel sequential querying for {m} samples.")
        sample_path = []
        
        sequential_prompts = []
        for i in range(m):
            if dataset_type == "Bernoulli":
                seq_prompt = (
                    f"Based on these observed {dataset_type} samples: " + ", ".join(map(str, observed_data[:10])) + "...\n"
                    f"Generate sample #{i+1} out of {m}. Output ONLY a single number, either 0 or 1. Don't explain:"
                )
            else:
                seq_prompt = (
                    f"Based on these observed {dataset_type} samples: " + ", ".join([f"{x:.2f}" for x in observed_data[:10]]) + "...\n"
                    f"Generate sample #{i+1} out of {m}. Output ONLY a single number (e.g., 1.25 or -0.34). Don't explain:"
                )
            sequential_prompts.append(seq_prompt)
        
        valid_values = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(get_single_sample, model_name, seq_prompt, dataset_type): idx
                             for idx, seq_prompt in enumerate(sequential_prompts)}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    value = future.result()
                    if value is not None:
                        valid_values.append((idx, value))
                except Exception as e:
                    print(f"Error getting sample for index {idx}: {e}")
        
        if len(valid_values) == m:
            valid_values.sort(key=lambda x: x[0])
            sample_path = [v for _, v in valid_values]
        else:
            print(f"Sequential mode only got {len(valid_values)}/{m} valid values")
            return []  # Failed to get a complete path
    
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
# Experiment Functions for Each Dataset (Parallelized)
# =====================================

def run_parallel_path_generation(model_name, observed_data, m, dataset_type, num_paths):
    """
    Generate multiple sample paths in parallel using ThreadPoolExecutor
    
    Parameters:
      model_name: LLM identifier
      observed_data: observed samples
      m: number of future samples to predict
      dataset_type: "Bernoulli" or "Gaussian"
      num_paths: number of paths to generate
      
    Returns:
      List of valid sample paths
    """
    valid_paths = []
    
    target_attempts = min(num_paths * 3, 150)
    
    print(f"Generating {num_paths} paths in parallel (attempting up to {target_attempts})...")
    
    tasks = [(model_name, observed_data, m, dataset_type) for _ in range(target_attempts)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_sample_path, *task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating paths"):
            try:
                path = future.result()
                if path and len(path) == m:
                    valid_paths.append(path)
                    if len(valid_paths) >= num_paths:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
            except Exception as e:
                print(f"Path generation error: {e}")
    
    if len(valid_paths) < num_paths:
        print(f"WARNING: Could only generate {len(valid_paths)}/{num_paths} valid paths")
    else:
        print(f"Successfully generated {len(valid_paths)} valid paths")
    
    return valid_paths[:num_paths]

def run_bernoulli_experiment(model_name, theta, n=BERNOULLI_n, m_factor=0.5, num_paths=NUM_PATHS):
    """
    Run a Bernoulli synthetic experiment with parallel path generation.
    
    Parameters:
      model_name: LLM identifier.
      theta: true Bernoulli parameter.
      n: number of observed samples.
      m_factor: multiplier for m relative to n.
      num_paths: number of Monte Carlo sample paths.
      
    Returns:
      T1, T2 diagnostic values and the list of sample paths.
    """
    m = int(n * m_factor)
    observed_data = np.random.binomial(1, theta, n).tolist()
    
    sample_paths = run_parallel_path_generation(model_name, observed_data, m, "Bernoulli", num_paths)
    
    T1 = compute_T1(sample_paths) if sample_paths else np.nan
    T2 = compute_T2(sample_paths, k=1) if sample_paths else np.nan
    return T1, T2, sample_paths

def run_gaussian_experiment(model_name, theta, n=GAUSSIAN_n, m_factor=0.5, num_paths=NUM_PATHS):
    """
    Run a Gaussian synthetic experiment with parallel path generation.
    
    Parameters:
      model_name: LLM identifier.
      theta: true mean of the Gaussian.
      n: number of observed samples.
      m_factor: multiplier for m relative to n.
      num_paths: number of Monte Carlo sample paths.
      
    Returns:
      T1, T2 diagnostic values and the list of sample paths.
    """
    m = int(n * m_factor)
    observed_data = np.random.normal(theta, 1, n).tolist()
    
    sample_paths = run_parallel_path_generation(model_name, observed_data, m, "Gaussian", num_paths)
    
    T1 = compute_T1(sample_paths) if sample_paths else np.nan
    T2 = compute_T2(sample_paths, k=1) if sample_paths else np.nan
    return T1, T2, sample_paths

# ===================
# Main Experiment Loop
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
    
    filename = f"results/{model_name}_results.json"
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Results saved to {filename}")

def plot_model_results(model_name, results_bernoulli, results_gaussian):
    """Create and save plots for a single model's results"""
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    x_vals = []
    T1_vals = []
    T2_vals = []
    
    for key, stats in results_bernoulli.items():
        parts = key.split('_')
        theta_val = float(parts[1])
        m_factor_val = float(parts[3])
        x_vals.append(f"θ = {theta_val}\nm = {int(BERNOULLI_n*m_factor_val)}")
        T1_vals.append(stats["T1"])
        T2_vals.append(stats["T2"])
    
    plt.plot(x_vals, T1_vals, marker='o', linestyle='-', label="T₁ (mean diff)")
    plt.plot(x_vals, T2_vals, marker='s', linestyle='--', label="T₂ (seq diff)")
    plt.title(f"{model_name} - Bernoulli Distribution")
    plt.xlabel("θ and forecast horizon (m)")
    plt.ylabel("Diagnostic Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_bernoulli.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    x_vals = []
    T1_vals = []
    T2_vals = []
    
    for key, stats in results_gaussian.items():
        parts = key.split('_')
        theta_val = float(parts[1])
        m_factor_val = float(parts[3])
        x_vals.append(f"θ = {theta_val}\nm = {int(GAUSSIAN_n*m_factor_val)}")
        T1_vals.append(stats["T1"])
        T2_vals.append(stats["T2"])
    
    plt.plot(x_vals, T1_vals, marker='o', linestyle='-', label="T₁ (mean diff)")
    plt.plot(x_vals, T2_vals, marker='s', linestyle='--', label="T₂ (seq diff)")
    plt.title(f"{model_name} - Gaussian Distribution")
    plt.xlabel("θ and forecast horizon (m)")
    plt.ylabel("Diagnostic Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_gaussian.png")
    plt.close()
    
    print(f"Plots saved for {model_name}")

def plot_comparison(results_bernoulli, results_gaussian):
    """Create comparison plots across all models after all experiments are complete"""
    if len(results_bernoulli) <= 1:
        return
    
    os.makedirs("plots", exist_ok=True)
    
    fig, axs = plt.subplots(1, len(models), figsize=(16, 4))
    if len(models) == 1:
        axs = [axs]
        
    for i, model in enumerate(models):
        if model not in results_bernoulli:
            continue
        x_vals = []
        T1_vals = []
        for key, stats in results_bernoulli[model].items():
            parts = key.split('_')
            theta_val = float(parts[1])
            m_factor_val = float(parts[3])
            x_vals.append(f"θ = {theta_val}\nm = {int(BERNOULLI_n*m_factor_val)}")
            T1_vals.append(stats["T1"])
        axs[i].plot(x_vals, T1_vals, marker='o', linestyle='-', label="T₁ (mean diff)")
        axs[i].set_title(f"{model} - Bernoulli")
        axs[i].set_xlabel("θ and m (n/mFactor)")
        axs[i].set_ylabel("Diagnostic T₁")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig("plots/comparison_bernoulli.png")
    plt.close()
    
    fig, axs = plt.subplots(1, len(models), figsize=(16, 4))
    if len(models) == 1:
        axs = [axs]
        
    for i, model in enumerate(models):
        if model not in results_gaussian:
            continue
        x_vals = []
        T1_vals = []
        for key, stats in results_gaussian[model].items():
            parts = key.split('_')
            theta_val = float(parts[1])
            m_factor_val = float(parts[3])
            x_vals.append(f"θ = {theta_val}\nm = {int(GAUSSIAN_n*m_factor_val)}")
            T1_vals.append(stats["T1"])
        axs[i].plot(x_vals, T1_vals, marker='o', linestyle='-', label="T₁ (mean diff)")
        axs[i].set_title(f"{model} - Gaussian")
        axs[i].set_xlabel("θ and m (n/mFactor)")
        axs[i].set_ylabel("Diagnostic T₁")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig("plots/comparison_gaussian.png")
    plt.close()
    
    print("Comparison plots saved")

def main():
    bernoulli_thetas = [0.3, 0.5, 0.7]
    gaussian_thetas = [-1, 0, 1]
    
    all_results_bernoulli = {}
    all_results_gaussian = {}
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Starting experiments for {model}")
        print(f"{'='*50}")
        
        results_bernoulli = {}
        results_gaussian = {}
        
        print(f"\nRunning Bernoulli experiments for {model}")
        for theta in bernoulli_thetas:
            for m_factor in m_factors:
                T1, T2, paths = run_bernoulli_experiment(model, theta, n=BERNOULLI_n,
                                                         m_factor=m_factor, num_paths=NUM_PATHS)
                key = f"theta_{theta}_mFactor_{m_factor}"
                results_bernoulli[key] = {"T1": T1, "T2": T2, "paths": paths}
                print(f"Bernoulli {model}, θ={theta}, mFactor={m_factor} => T₁: {T1:.3f}, T₂: {T2:.3f}")
        
        print(f"\nRunning Gaussian experiments for {model}")
        for theta in gaussian_thetas:
            for m_factor in m_factors:
                T1, T2, paths = run_gaussian_experiment(model, theta, n=GAUSSIAN_n,
                                                        m_factor=m_factor, num_paths=NUM_PATHS)
                key = f"theta_{theta}_mFactor_{m_factor}"
                results_gaussian[key] = {"T1": T1, "T2": T2, "paths": paths}
                print(f"Gaussian {model}, θ={theta}, mFactor={m_factor} => T₁: {T1:.3f}, T₂: {T2:.3f}")
        
        save_results_to_file(model, results_bernoulli, results_gaussian)
        plot_model_results(model, results_bernoulli, results_gaussian)
        
        all_results_bernoulli[model] = results_bernoulli
        all_results_gaussian[model] = results_gaussian
        
        print(f"\n{'='*50}")
        print(f"Finished experiments for {model}")
        print(f"{'='*50}")
    
    plot_comparison(all_results_bernoulli, all_results_gaussian)

if __name__ == "__main__":
    main()

