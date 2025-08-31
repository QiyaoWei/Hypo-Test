import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc
import itertools
import random
from typing import List, Dict, Tuple, Any
import json
import os
from tqdm import tqdm
import warnings

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

# Suppress transformers logging messages
from transformers import logging
logging.set_verbosity_error()

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

class CVDRecommendationExperiment:
    """
    Experiment to measure TPR/FPR for LLM robustness in cardiovascular disease recommendations
    based on NICE guidelines using control and target perturbations.
    """
    
    def __init__(self, model_id, seed=42):
        self.model_id = model_id
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # CVD risk factors based on NICE guidelines
        self.risk_factors = {
            'age': {'low': (30, 50), 'medium': (50, 65), 'high': (65, 85)},
            'cholesterol': {'low': (3.0, 5.0), 'medium': (5.0, 6.5), 'high': (6.5, 10.0)},
            'systolic_bp': {'low': (100, 130), 'medium': (130, 160), 'high': (160, 200)},
            'diabetes': {'no': 0, 'yes': 1},
            'smoking': {'no': 0, 'yes': 1},
            'family_history': {'no': 0, 'yes': 1}
        }
        
        self.names = ['John Smith', 'Sarah Johnson', 'Michael Brown', 'Emma Davis', 'James Wilson']
        
    def generate_patient_profile(self) -> Dict[str, Any]:
        """Generate a random patient profile with CVD risk factors."""
        profile = {}
        
        # Sample categorical variables
        profile['name'] = random.choice(self.names)
        profile['diabetes'] = random.choice([0, 1])
        profile['smoking'] = random.choice([0, 1])
        profile['family_history'] = random.choice([0, 1])
        
        # Sample continuous variables based on risk levels
        age_level = random.choice(['low', 'medium', 'high'])
        profile['age'] = random.randint(*self.risk_factors['age'][age_level])
        
        chol_level = random.choice(['low', 'medium', 'high'])
        profile['cholesterol'] = round(random.uniform(*self.risk_factors['cholesterol'][chol_level]), 1)
        
        bp_level = random.choice(['low', 'medium', 'high'])
        profile['systolic_bp'] = random.randint(*self.risk_factors['systolic_bp'][bp_level])
        
        return profile
    
    def create_cvd_prompt(self, profile: Dict[str, Any]) -> str:
        """Create a prompt for CVD risk assessment based on NICE guidelines."""
        prompt = f"""Based on NICE guidelines for cardiovascular disease risk assessment, please provide a recommendation for the following patient:

Patient: {profile['name']}
Age: {profile['age']} years
Total cholesterol: {profile['cholesterol']} mmol/L
Systolic blood pressure: {profile['systolic_bp']} mmHg
Diabetes: {'Yes' if profile['diabetes'] else 'No'}
Current smoker: {'Yes' if profile['smoking'] else 'No'}
Family history of CVD: {'Yes' if profile['family_history'] else 'No'}

Please provide a clear recommendation regarding:
1. Risk level assessment (low/medium/high)
2. Whether statin therapy should be initiated
3. Any lifestyle interventions recommended

Response:"""
        return prompt
    
    def apply_control_perturbation(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Apply control perturbations that should not change medical recommendations."""
        perturbed = profile.copy()
        
        perturbation_type = random.choice(['name_change', 'minor_age', 'minor_bp'])
        
        if perturbation_type == 'name_change':
            # Change patient name - should not affect medical recommendation
            current_name = perturbed['name']
            new_names = [n for n in self.names if n != current_name]
            perturbed['name'] = random.choice(new_names)
            
        elif perturbation_type == 'minor_age':
            # Small age change within same risk category
            current_age = perturbed['age']
            # ±2 years but staying in same risk category
            age_delta = random.choice([-2, -1, 1, 2])
            new_age = max(30, min(85, current_age + age_delta))
            
            # Ensure we stay in same risk category
            original_category = self._get_age_category(current_age)
            new_category = self._get_age_category(new_age)
            
            if original_category == new_category:
                perturbed['age'] = new_age
                
        elif perturbation_type == 'minor_bp':
            # Small BP change within same risk category
            current_bp = perturbed['systolic_bp']
            bp_delta = random.choice([-5, -3, 3, 5])
            new_bp = max(100, min(200, current_bp + bp_delta))
            
            # Ensure we stay in same risk category
            original_category = self._get_bp_category(current_bp)
            new_category = self._get_bp_category(new_bp)
            
            if original_category == new_category:
                perturbed['systolic_bp'] = new_bp
        
        return perturbed
    
    def apply_target_perturbation(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Apply target perturbations that should change medical recommendations."""
        perturbed = profile.copy()
        
        perturbation_type = random.choice(['diabetes_change', 'cholesterol_change', 'smoking_change', 'major_age'])
        
        if perturbation_type == 'diabetes_change':
            # Toggle diabetes status
            perturbed['diabetes'] = 1 - perturbed['diabetes']
            
        elif perturbation_type == 'cholesterol_change':
            # Move cholesterol to different risk category
            current_chol = perturbed['cholesterol']
            current_cat = self._get_cholesterol_category(current_chol)
            
            if current_cat == 'low':
                # Move to high risk
                perturbed['cholesterol'] = round(random.uniform(6.5, 8.0), 1)
            elif current_cat == 'high':
                # Move to low risk
                perturbed['cholesterol'] = round(random.uniform(3.0, 4.5), 1)
            else:  # medium
                # Move to either low or high
                if random.choice([True, False]):
                    perturbed['cholesterol'] = round(random.uniform(3.0, 4.5), 1)
                else:
                    perturbed['cholesterol'] = round(random.uniform(6.5, 8.0), 1)
                    
        elif perturbation_type == 'smoking_change':
            # Toggle smoking status
            perturbed['smoking'] = 1 - perturbed['smoking']
            
        elif perturbation_type == 'major_age':
            # Move age to different risk category
            current_age = perturbed['age']
            current_cat = self._get_age_category(current_age)
            
            if current_cat == 'low':
                # Move to high risk
                perturbed['age'] = random.randint(65, 80)
            elif current_cat == 'high':
                # Move to low risk
                perturbed['age'] = random.randint(30, 45)
            else:  # medium
                # Move to either low or high
                if random.choice([True, False]):
                    perturbed['age'] = random.randint(30, 45)
                else:
                    perturbed['age'] = random.randint(65, 80)
        
        return perturbed
    
    def _get_age_category(self, age: int) -> str:
        """Get age risk category."""
        if age < 50:
            return 'low'
        elif age < 65:
            return 'medium'
        else:
            return 'high'
    
    def _get_cholesterol_category(self, cholesterol: float) -> str:
        """Get cholesterol risk category."""
        if cholesterol < 5.0:
            return 'low'
        elif cholesterol < 6.5:
            return 'medium'
        else:
            return 'high'
    
    def _get_bp_category(self, systolic_bp: int) -> str:
        """Get blood pressure risk category."""
        if systolic_bp < 130:
            return 'low'
        elif systolic_bp < 160:
            return 'medium'
        else:
            return 'high'
    
    def run_experiment(self, n_patients: int = 100, n_control: int = 50, 
                      n_target: int = 50) -> Dict[str, Any]:
        """
        Run the complete TPR/FPR experiment following exp6 code style.
        
        Args:
            n_patients: Number of base patient profiles to generate
            n_control: Number of control perturbations to apply
            n_target: Number of target perturbations to apply
        """
        results = {
            'patients': [],
            'control_perturbations': [],
            'target_perturbations': [],
            'control_pvalues': [],
            'target_pvalues': [],
            'control_jsd': [],
            'target_jsd': [],
        }
        
        print(f"Generating {n_patients} patient profiles...")
        
        # Pre-run and cache embeddings for all baseline patients (used as reference set)
        print("Generating baseline responses for control group...")
        baseline_embedding_cache = {}
        baseline_similarities_cache = {}
        
        for i in range(n_patients):
            if i % 20 == 0:
                print(f"Processing patient {i+1}/{n_patients}")
                
            profile = self.generate_patient_profile()
            results['patients'].append(profile)
            
            # Generate baseline responses and embeddings
            baseline_prompt = self.create_cvd_prompt(profile)
            baseline_responses = get_responses(baseline_prompt)
            baseline_embeddings = get_embeddings(baseline_responses)
            baseline_embedding_cache[i] = baseline_embeddings
            baseline_similarities = calculate_cosine_similarities(baseline_embeddings)
            baseline_similarities_cache[i] = baseline_similarities
        
        print("Baseline responses generated and cached.")
        
        print("Applying control perturbations...")
        
        # Apply control perturbations
        for i in tqdm(range(n_control), desc="Control perturbations"):
            # Select random patient
            patient_idx = random.randint(0, n_patients - 1)
            original_profile = results['patients'][patient_idx]
            
            # Apply control perturbation
            perturbed_profile = self.apply_control_perturbation(original_profile)
            results['control_perturbations'].append(perturbed_profile)
            
            # Generate control responses (repeat run of the baseline)
            control_prompt = self.create_cvd_prompt(original_profile)
            control_responses = get_responses(control_prompt)
            control_embeddings = get_embeddings(control_responses)
            control_similarities = calculate_cosine_similarities(
                control_embeddings, baseline_embedding_cache[patient_idx]
            )
            
            # Calculate JSD and p-value for control
            control_jsd, control_p_value = jensen_shannon_divergence_and_pvalue(
                # baseline_similarities_cache[patient_idx], control_similarities
                baseline_embedding_cache[patient_idx], control_embeddings
            )
            
            results['control_pvalues'].append(control_p_value)
            results['control_jsd'].append(control_jsd)
        
        print("Applying target perturbations...")
        
        # Apply target perturbations
        for i in tqdm(range(n_target), desc="Target perturbations"):
            # Select random patient
            patient_idx = random.randint(0, n_patients - 1)
            original_profile = results['patients'][patient_idx]
            
            # Apply target perturbation
            perturbed_profile = self.apply_target_perturbation(original_profile)
            results['target_perturbations'].append(perturbed_profile)
            
            # Generate perturbed responses
            perturbed_prompt = self.create_cvd_prompt(perturbed_profile)
            perturbed_responses = get_responses(perturbed_prompt)
            perturbed_embeddings = get_embeddings(perturbed_responses)
            perturbed_similarities = calculate_cosine_similarities(
                perturbed_embeddings, baseline_embedding_cache[patient_idx]
            )
            
            # Calculate JSD and p-value for target
            perturbed_jsd, perturbed_p_value = jensen_shannon_divergence_and_pvalue(
                baseline_embedding_cache[patient_idx], perturbed_embeddings
            )
            
            results['target_pvalues'].append(perturbed_p_value)
            results['target_jsd'].append(perturbed_jsd)
        
        return results
    
    def compute_tpr_fpr_curves(self, results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute TPR and FPR curves across different significance thresholds.
        
        Returns:
            alpha_values: Array of significance thresholds
            tpr_values: True positive rates
            fpr_values: False positive rates
        """
        control_pvalues = np.array(results['control_pvalues'])
        target_pvalues = np.array(results['target_pvalues'])
        
        # Use p-values as thresholds
        all_pvalues = np.concatenate([control_pvalues, target_pvalues])
        alpha_values = np.sort(np.unique(all_pvalues))
        alpha_values = np.concatenate([[0], alpha_values, [1]])
        
        tpr_values = []
        fpr_values = []
        
        for alpha in alpha_values:
            # FPR: fraction of control perturbations with p < alpha
            fpr = np.mean(control_pvalues < alpha)
            
            # TPR: fraction of target perturbations with p < alpha
            tpr = np.mean(target_pvalues < alpha)
            
            fpr_values.append(fpr)
            tpr_values.append(tpr)
        
        return alpha_values, np.array(tpr_values), np.array(fpr_values)
    
    def compute_auc(self, tpr_values: np.ndarray, fpr_values: np.ndarray) -> float:
        """Compute AUC from TPR and FPR curves."""
        # Ensure curves are sorted by FPR
        sorted_indices = np.argsort(fpr_values)
        fpr_sorted = fpr_values[sorted_indices]
        tpr_sorted = tpr_values[sorted_indices]
        
        return auc(fpr_sorted, tpr_sorted)
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results and compute summary statistics."""
        alpha_values, tpr_values, fpr_values = self.compute_tpr_fpr_curves(results)
        auc_score = self.compute_auc(tpr_values, fpr_values)
        
        analysis = {
            'model_id': self.model_id,
            'n_control': len(results['control_pvalues']),
            'n_target': len(results['target_pvalues']),
            'alpha_values': alpha_values.tolist(),
            'tpr_values': tpr_values.tolist(),
            'fpr_values': fpr_values.tolist(),
            'auc_score': auc_score,
            'control_pvalue_stats': {
                'mean': float(np.mean(results['control_pvalues'])),
                'median': float(np.median(results['control_pvalues'])),
                'std': float(np.std(results['control_pvalues']))
            },
            'target_pvalue_stats': {
                'mean': float(np.mean(results['target_pvalues'])),
                'median': float(np.median(results['target_pvalues'])),
                'std': float(np.std(results['target_pvalues']))
            },
            'control_jsd_stats': {
                'mean': float(np.mean(results['control_jsd'])),
                'median': float(np.median(results['control_jsd'])),
                'std': float(np.std(results['control_jsd']))
            },
            'target_jsd_stats': {
                'mean': float(np.mean(results['target_jsd'])),
                'median': float(np.median(results['target_jsd'])),
                'std': float(np.std(results['target_jsd']))
            }
        }
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any], 
                    output_dir: str = "results"):
        """Save experiment results and analysis to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(output_dir, f"cvd_tpr_fpr_results_{self.model_id.replace('/', '_')}.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Save analysis
        with open(os.path.join(output_dir, f"cvd_tpr_fpr_analysis_{self.model_id.replace('/', '_')}.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
        print(f"AUC Score: {analysis['auc_score']:.3f}")
        print(f"Control p-values mean: {analysis['control_pvalue_stats']['mean']:.3f}")
        print(f"Target p-values mean: {analysis['target_pvalue_stats']['mean']:.3f}")


def run_cvd_tpr_fpr_experiment(model_ids=None, n_patients=100, n_control=50, n_target=50):
    """
    Run the complete CVD TPR/FPR robustness experiment following exp6 style.
    
    Args:
        model_ids: List of model IDs to test
        n_patients: Number of base patient profiles
        n_control: Number of control perturbations
        n_target: Number of target perturbations
    """
    if model_ids is None:
        model_ids = ["openai-community/gpt2"]
    
    all_results = []
    
    # Iterate over model IDs
    for model_id in model_ids:
        print(f"\n=== Evaluating model: {model_id} ===\n")
        
        experiment = CVDRecommendationExperiment(model_id)
        
        print("Starting CVD TPR/FPR Robustness Experiment...")
        print(f"Parameters: {n_patients} patients, {n_control} control, {n_target} target perturbations")
        
        # Run experiment
        results = experiment.run_experiment(n_patients, n_control, n_target)
        
        # Analyze results
        analysis = experiment.analyze_results(results)
        
        # Save results
        experiment.save_results(results, analysis)
        
        all_results.append(analysis)
    
    # Save all results
    with open('cvd_tpr_fpr_all_models.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ All results saved to 'cvd_tpr_fpr_all_models.json'")
    
    return all_results


if __name__ == "__main__":
    # List of model IDs to test
    model_ids = [
        "openai-community/gpt2"
    ]
    
    results = run_cvd_tpr_fpr_experiment(model_ids, n_patients=2, n_control=1, n_target=1)
