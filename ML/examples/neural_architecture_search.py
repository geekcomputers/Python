import sys
sys.path.insert(0, '.')

import torch
from src.python.neuralforge.nas.search_space import SearchSpace
from src.python.neuralforge.nas.evolution import EvolutionarySearch
from src.python.neuralforge.nas.evaluator import ProxyEvaluator
from src.python.neuralforge.data.dataset import SyntheticDataset, DataLoaderBuilder
from src.python.neuralforge.config import Config

def main():
    config = Config()
    config.nas_enabled = True
    config.nas_population_size = 15
    config.nas_generations = 20
    config.nas_mutation_rate = 0.15
    
    search_config = {
        'num_layers': 15,
        'num_blocks': 4
    }
    
    search_space = SearchSpace(search_config)
    
    train_dataset = SyntheticDataset(num_samples=1000, num_classes=10)
    val_dataset = SyntheticDataset(num_samples=200, num_classes=10)
    
    loader_builder = DataLoaderBuilder(config)
    train_loader = loader_builder.build_train_loader(train_dataset)
    val_loader = loader_builder.build_val_loader(val_dataset)
    
    evaluator = ProxyEvaluator(device=config.device)
    
    evolution = EvolutionarySearch(
        search_space=search_space,
        evaluator=evaluator,
        population_size=config.nas_population_size,
        generations=config.nas_generations,
        mutation_rate=config.nas_mutation_rate
    )
    
    print("Starting Neural Architecture Search...")
    best_architecture = evolution.search()
    
    print(f"\nBest Architecture Found:")
    print(f"Fitness: {best_architecture.fitness:.4f}")
    print(f"Accuracy: {best_architecture.accuracy:.2f}%")
    print(f"Parameters: {best_architecture.params:,}")
    print(f"FLOPs: {best_architecture.flops:,}")
    
    print("\nTop 5 Architectures:")
    top_k = evolution.get_top_k_architectures(k=5)
    for i, arch in enumerate(top_k, 1):
        print(f"{i}. Fitness: {arch.fitness:.4f}, Acc: {arch.accuracy:.2f}%, Params: {arch.params:,}")
    
    model = search_space.build_model(best_architecture, num_classes=10)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

if __name__ == '__main__':
    main()
