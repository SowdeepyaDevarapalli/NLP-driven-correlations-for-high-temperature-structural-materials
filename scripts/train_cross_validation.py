#!/usr/bin/env python3

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpERTCrossFoldTrainer:
    def __init__(self, base_dir, spert_dir, epochs: int | None = None):
        self.base_dir = Path(base_dir)
        self.spert_dir = Path(spert_dir)
        self.data_dir = self.base_dir / "data"
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.epochs_override = epochs
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)
        
    def train_fold(self, fold_num):
        """Train SpERT on a specific fold."""
        logger.info(f"Training fold {fold_num}")
        
        fold_dir = self.data_dir / f"fold_{fold_num}"
        train_path = fold_dir / "train.json"
        valid_path = fold_dir / "valid.json"
        
        if not train_path.exists() or not valid_path.exists():
            logger.error(f"Data files not found for fold {fold_num}")
            return False
        
        # Create model directory for this fold
        model_dir = self.models_dir / f"fold_{fold_num}"
        model_dir.mkdir(exist_ok=True)
        
        # Update config file with fold-specific paths (using absolute paths)
        # Determine epochs for this run
        epochs_val = self.epochs_override if self.epochs_override is not None else 20

        config_lines = [
            f"label = materials_spert_fold_{fold_num}",
            "model_type = spert",
            "model_path = bert-base-cased",
            "tokenizer_path = bert-base-cased",
            f"train_path = {str(train_path.absolute())}",
            f"valid_path = {str(valid_path.absolute())}",
            f"types_path = {str((self.configs_dir / 'types.json').absolute())}",
            f"save_path = {str(model_dir.absolute())}",
            f"log_path = {str(model_dir.absolute())}",
            "neg_entity_count = 100",
            "neg_relation_count = 100",
            f"epochs = {epochs_val}",
            "lr = 5e-5",
            "lr_warmup = 0.1",
            "weight_decay = 0.01",
            "max_grad_norm = 1.0",
            "rel_filter_threshold = 0.4",
            "size_embedding = 25",
            "prop_drop = 0.1",
            "max_span_size = 10",
            "train_batch_size = 1",
            "eval_batch_size = 1",
            "max_pairs = 1000",
            "sampling_processes = 4",
            "store_predictions = true",
            "store_examples = true",
            "lowercase = false"
        ]
        config_content = "\n".join(config_lines)
        
        fold_config_path = self.configs_dir / f"train_config_fold_{fold_num}.conf"
        with open(fold_config_path, 'w') as f:
            f.write(config_content)
        
        # Fixed: Use correct SpERT script path
        spert_script = self.spert_dir / "spert.py"
        
        if not spert_script.exists():
            logger.error(f"SpERT script not found at: {spert_script}")
            return False
        
        # Run SpERT training
        cmd = [
            sys.executable,
            str(spert_script),
            "train",
            "--config", str(fold_config_path)
        ]
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            # Run from the base directory, not spert directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir))
            
            if result.returncode != 0:
                logger.error(f"Training failed for fold {fold_num}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
            else:
                logger.info(f"Training completed for fold {fold_num}")
                # Try to parse the exact save path from stdout to avoid race with concurrent runs
                run_dir = None
                m = re.search(r"Saved in:\s*(.+)", result.stdout or "")
                if m:
                    # The save path printed is models/.../<label>/<timestamp>
                    candidate = Path(m.group(1).strip())
                    if candidate.exists():
                        run_dir = candidate
                # Fallback: pick latest timestamp under the label
                if run_dir is None:
                    label_dir = model_dir / f"materials_spert_fold_{fold_num}"
                    candidates = sorted([p for p in label_dir.iterdir() if p.is_dir()],
                                        key=lambda p: p.stat().st_ctime, reverse=True)
                    run_dir = candidates[0] if candidates else None
                if run_dir is None:
                    logger.warning(f"Could not determine run directory for fold {fold_num}; evaluation may fail.")
                return run_dir
                
        except Exception as e:
            logger.error(f"Error running training for fold {fold_num}: {e}")
            return False
    
    def evaluate_fold(self, fold_num, run_dir: Path | None = None):
        """Evaluate SpERT on a specific fold."""
        logger.info(f"Evaluating fold {fold_num}")
        
        fold_dir = self.data_dir / f"fold_{fold_num}"
        valid_path = fold_dir / "valid.json"
        model_dir = self.models_dir / f"fold_{fold_num}"
        # The actual save path is models/fold_X/<label>/<timestamp>/final_model
        # Prefer the run_dir returned by training to avoid selecting a different, concurrently running timestamp
        if run_dir is not None:
            latest_run = Path(run_dir)
        else:
            label_dir = model_dir / f"materials_spert_fold_{fold_num}"
            if not label_dir.exists():
                logger.error(f"No label directory found for fold {fold_num} at {label_dir}")
                return False
            candidates = sorted([p for p in label_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_ctime, reverse=True)
            if not candidates:
                logger.error(f"No run directories found under {label_dir}")
                return False
            latest_run = candidates[0]
        model_path = latest_run / "final_model"
        if not model_path.exists():
            # fallback to any model_*/best dirs
            alt = list(latest_run.glob("model_*/")) + list(latest_run.glob("*best/"))
            if alt:
                model_path = alt[0]
            else:
                logger.error(f"No trained model directory found in {latest_run}")
                return False
        
        # Create evaluation config
        eval_config_lines = [
            f"label = materials_spert_fold_{fold_num}",
            "model_type = spert",
            f"model_path = {str(model_path)}",
            "tokenizer_path = bert-base-cased",
            "rel_filter_threshold = 0.4",
            "max_span_size = 10",
            "eval_batch_size = 1",
            "max_pairs = 1000",
            "store_predictions = true",
            "store_examples = true",
            "lowercase = false",
            "sampling_processes = 4",
            f"log_path = {str(self.results_dir / f'fold_{fold_num}')}",
            f"types_path = {str(self.configs_dir / 'types.json')}",
            f"dataset_path = {str(valid_path)}"
        ]
        eval_config_content = "\n".join(eval_config_lines)
        
        eval_config_path = self.configs_dir / f"eval_config_fold_{fold_num}.conf"
        with open(eval_config_path, 'w') as f:
            f.write(eval_config_content)
        
        # Create results directory for this fold
        fold_results_dir = self.results_dir / f"fold_{fold_num}"
        fold_results_dir.mkdir(exist_ok=True)
        
        # Fixed: Use correct SpERT script path
        spert_script = self.spert_dir / "spert.py"
        
        # Run SpERT evaluation
        cmd = [
            sys.executable,
            str(spert_script),
            "eval",
            "--config", str(eval_config_path)
        ]
        
        try:
            logger.info(f"Running evaluation: {' '.join(cmd)}")
            # Run from the base directory, not spert directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir))

            # Persist stdout/stderr so printed per-type tables are available
            try:
                (fold_results_dir / "stdout.txt").write_text(result.stdout or "")
                (fold_results_dir / "stderr.txt").write_text(result.stderr or "")
            except Exception as io_err:
                logger.warning(f"Could not write eval stdout/stderr for fold {fold_num}: {io_err}")

            if result.returncode != 0:
                logger.error(f"Evaluation failed for fold {fold_num}")
                return False
            else:
                logger.info(f"Evaluation completed for fold {fold_num}")
                return True

        except Exception as e:
            logger.error(f"Error running evaluation for fold {fold_num}: {e}")
            return False
    
    def run_cross_validation(self, n_folds=5):
        """Run complete cross-validation training and evaluation."""
        logger.info(f"Starting {n_folds}-fold cross-validation")
        
        results = {}
        
        for fold in range(n_folds):
            logger.info(f"Processing fold {fold}")
            
            # Train on this fold
            run_dir = self.train_fold(fold)
            if run_dir:
                # Evaluate on this fold
                if self.evaluate_fold(fold, run_dir=run_dir):
                    results[fold] = "success"
                else:
                    results[fold] = "eval_failed"
            else:
                results[fold] = "train_failed"
        
        # Save results summary
        results_summary_path = self.results_dir / "cross_validation_summary.json"
        with open(results_summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Cross-validation completed. Results: {results}")
        return results

def main():
    parser = argparse.ArgumentParser(description="SpERT Cross-Fold Validation Training")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing data, configs, etc.")
    parser.add_argument("--spert_dir", type=str, required=True,
                       help="Directory containing SpERT source code")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count in generated train configs (default 20)")
    
    args = parser.parse_args()
    
    trainer = SpERTCrossFoldTrainer(args.base_dir, args.spert_dir, epochs=args.epochs)
    results = trainer.run_cross_validation(args.n_folds)
    
    print(f"Cross-validation results: {results}")

if __name__ == "__main__":
    main()