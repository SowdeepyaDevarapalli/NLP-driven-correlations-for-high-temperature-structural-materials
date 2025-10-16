#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_spert(base_dir):
    """Download and setup SpERT repository."""
    base_path = Path(base_dir)
    spert_dir = base_path / "spert"
    
    if spert_dir.exists():
        logger.info("SpERT directory already exists. Skipping download.")
        return str(spert_dir)
    
    logger.info("Cloning SpERT repository...")
    
    try:
        # Clone the SpERT repository
        cmd = ["git", "clone", "https://github.com/lavis-nlp/spert.git", str(spert_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base_path))
        
        if result.returncode != 0:
            logger.error(f"Failed to clone SpERT repository: {result.stderr}")
            return None
        
        logger.info("SpERT repository cloned successfully")
        
        # Install SpERT requirements
        requirements_file = spert_dir / "requirements.txt"
        if requirements_file.exists():
            logger.info("Installing SpERT requirements...")
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install requirements: {result.stderr}")
            else:
                logger.info("SpERT requirements installed successfully")
        
        return str(spert_dir)
        
    except Exception as e:
        logger.error(f"Error setting up SpERT: {e}")
        return None

def main():
    base_dir = Path(__file__).parent.parent
    spert_dir = setup_spert(base_dir)
    
    if spert_dir:
        print(f"SpERT setup complete. Directory: {spert_dir}")
        print("\nNext steps:")
        print("1. Run cross-validation training:")
        print(f"   python scripts/train_cross_validation.py --base_dir {base_dir} --spert_dir {spert_dir}")
        print("2. Evaluate results:")
        print(f"   python scripts/evaluate_cross_validation.py --results_dir {base_dir}/results")
    else:
        print("SpERT setup failed")

if __name__ == "__main__":
    main()