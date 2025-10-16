import json
import random
import os
from sklearn.model_selection import KFold
import numpy as np

def create_cross_fold_splits(data_path, output_dir, n_folds=5, random_seed=42):
    """
    Create cross-fold validation splits for SpERT training.
    
    Args:
        data_path: Path to the spert.json file
        output_dir: Directory to save fold splits
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load the data
    with open(data_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers from {data_path}")
    
    # Create indices for papers
    paper_indices = list(range(len(papers)))
    random.shuffle(paper_indices)
    
    # Create KFold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Create fold directories
    for fold in range(n_folds):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
    
    # Generate splits
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(paper_indices)):
        print(f"\nFold {fold_idx}: Train={len(train_indices)}, Val={len(val_indices)}")
        
        # Get actual paper indices
        train_papers = [papers[paper_indices[i]] for i in train_indices]
        val_papers = [papers[paper_indices[i]] for i in val_indices]
        
        # Save train set
        train_path = os.path.join(output_dir, f"fold_{fold_idx}", "train.json")
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_papers, f, indent=2, ensure_ascii=False)
        
        # Save validation set
        val_path = os.path.join(output_dir, f"fold_{fold_idx}", "valid.json")
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_papers, f, indent=2, ensure_ascii=False)
        
        # Print statistics for this fold
        train_entities = sum(len(paper['entities']) for paper in train_papers)
        train_relations = sum(len(paper['relations']) for paper in train_papers)
        val_entities = sum(len(paper['entities']) for paper in val_papers)
        val_relations = sum(len(paper['relations']) for paper in val_papers)
        
        print(f"  Train: {train_entities} entities, {train_relations} relations")
        print(f"  Val: {val_entities} entities, {val_relations} relations")

def analyze_dataset_statistics(data_path):
    """Analyze the dataset to understand entity and relation distributions."""
    with open(data_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    entity_types = {}
    relation_types = {}
    total_entities = 0
    total_relations = 0
    
    for paper in papers:
        total_entities += len(paper['entities'])
        total_relations += len(paper['relations'])
        
        for entity in paper['entities']:
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        for relation in paper['relations']:
            relation_type = relation['type']
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
    
    print("=== Dataset Statistics ===")
    print(f"Total papers: {len(papers)}")
    print(f"Total entities: {total_entities}")
    print(f"Total relations: {total_relations}")
    print(f"Avg entities per paper: {total_entities/len(papers):.2f}")
    print(f"Avg relations per paper: {total_relations/len(papers):.2f}")
    
    print("\n=== Entity Type Distribution ===")
    for entity_type, count in sorted(entity_types.items()):
        print(f"{entity_type}: {count} ({count/total_entities*100:.1f}%)")
    
    print("\n=== Relation Type Distribution ===")
    for relation_type, count in sorted(relation_types.items()):
        print(f"{relation_type}: {count} ({count/total_relations*100:.1f}%)")
    
    return entity_types, relation_types

if __name__ == "__main__":
    data_path = r"c:\Users\devar\OneDrive\Desktop\superman\spert.json"
    output_dir = r"c:\Users\devar\OneDrive\Desktop\superman\spert_materials\data"
    
    # Analyze dataset first
    entity_types, relation_types = analyze_dataset_statistics(data_path)
    
    # Create cross-fold splits
    create_cross_fold_splits(data_path, output_dir, n_folds=5)
    
    print("\n=== Cross-fold splits created successfully! ===")