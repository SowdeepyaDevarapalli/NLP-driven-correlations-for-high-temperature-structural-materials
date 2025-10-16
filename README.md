# SPERT Materials Science Information Extraction

This repository contains the complete implementation of SPERT (Span-based Entity and Relation Transformer) for materials science literature mining. This is the **20-paper best-performing model** achieving 39.1% entity F1 and 18.1% relation F1 scores.

## 📁 Repository Structure

```
spert_materials_clean/
├── spert/                    # Core SPERT framework
│   ├── args.py              # Command line argument parsing
│   ├── config_reader.py     # Configuration file handling
│   ├── spert.py            # Main SPERT entry point
│   ├── requirements.txt     # Python dependencies
│   └── spert/              # Core model implementation
│       ├── entities.py      # Entity extraction logic
│       ├── models.py        # Neural network architectures
│       ├── trainer.py       # Training orchestration
│       ├── evaluator.py     # Model evaluation
│       └── [other core files]
├── configs/                  # Configuration files
│   ├── train_config.conf    # Training configuration
│   ├── eval_config.conf     # Evaluation configuration
│   ├── test_config.conf     # Testing configuration
│   ├── types.conf           # Entity/relation type definitions
│   └── types.json           # Type mappings
├── scripts/                  # Training and utility scripts
│   ├── train_cross_validation.py  # Main training script
│   ├── create_folds.py      # Data preparation for CV
│   └── setup_spert.py       # Environment setup
├── data/                     # Training data (5-fold CV structure)
│   ├── fold_0/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
└── spert.json               # Root configuration
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r spert/requirements.txt

# Setup SPERT environment
python scripts/setup_spert.py
```

### 2. Training
```bash
# Run 5-fold cross-validation training
python scripts/train_cross_validation.py
```

### 3. Data Preparation (if needed)
```bash
# Create new data folds from raw data
python scripts/create_folds.py
```

## 📊 Model Performance

| Metric | Entity Extraction | Relation Extraction |
|--------|------------------|-------------------|
| **F1 Score** | 39.12% | 18.12% |
| **Precision** | 30.45% | 19.93% |
| **Recall** | 61.91% | 21.12% |
| **Accuracy** | 76.75% | 87.45% |

## 🏷️ Supported Entity Types

- **MATERIAL**: Chemical compounds, alloys, materials
- **PROPERTY**: Physical/chemical properties
- **VALUE**: Numerical values
- **UNIT**: Measurement units
- **PROCESS**: Manufacturing/synthesis processes
- **STRUCTURE**: Crystal structures, phases
- **CONDITION**: Temperature, pressure conditions
- And 5 additional specialized types

## 🔗 Supported Relation Types

- **HAS_PROPERTY**: Material-property relationships
- **HAS_VALUE**: Property-value relationships  
- **HAS_UNIT**: Value-unit relationships
- **OBSERVED_AT**: Measurement context relationships
- **HAS_STRUCTURE**: Material-structure relationships
- And 12 additional relationship types

## 💻 System Requirements

- **GPU**: Recommended for training (tested on Google Colab T4)
- **RAM**: 8GB+ recommended
- **Python**: 3.7+
- **CUDA**: Compatible GPU drivers if using local GPU

## 📖 Usage Examples

### Training from Scratch
```bash
python spert/spert.py train --config configs/train_config.conf
```

### Evaluation
```bash
python spert/spert.py eval --config configs/eval_config.conf
```

### Prediction on New Data
```bash
python spert/spert.py predict --config configs/test_config.conf
```

## 🔬 Research Context

This implementation represents the best-performing configuration from experiments comparing:
- **10-paper dataset**: Entity-only extraction (relations fail completely)
- **15-paper dataset**: Minimum viable joint extraction (11.1% relation F1)
- **20-paper dataset**: Optimal performance for available data (18.1% relation F1)

## 📚 Key Findings

1. **Critical Dataset Threshold**: Minimum 15 papers required for relation extraction
2. **Joint Learning Complexity**: Relations require significantly more data than entities
3. **Performance Scaling**: Gradual entity improvement vs. threshold effect for relations
4. **Production Readiness**: Current model suitable for research, needs 50+ papers for production

## 🛠️ Configuration

Main configuration files:
- `spert.json`: Root configuration
- `configs/train_config.conf`: Training hyperparameters
- `configs/types.conf`: Entity/relation type definitions

## 📄 Citation

If you use this code, please cite the original SPERT paper:
```
@inproceedings{eberts-ulges-2020-span,
    title = "Span-based Joint Entity and Relation Extraction with Transformer Pre-training",
    author = "Eberts, Markus and Ulges, Adrian",
    booktitle = "Proceedings of the 24th European Conference on Artificial Intelligence",
    year = "2020"
}
```

## 📧 Contact

For questions about this implementation or the materials science application, please open an issue in this repository.

---

**Note**: This is the production-ready codebase containing only essential files for training and inference. All experimental analysis files have been excluded for clarity.