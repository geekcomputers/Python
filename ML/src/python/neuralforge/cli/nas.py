import argparse
import torch
from neuralforge.nas.search_space import SearchSpace
from neuralforge.nas.evolution import EvolutionarySearch
from neuralforge.nas.evaluator import ProxyEvaluator
from neuralforge.data.datasets import get_dataset
from neuralforge.data.dataset import SyntheticDataset, DataLoaderBuilder
from neuralforge.config import Config

def main():
    parser = argparse.ArgumentParser(
        description='NeuralForge - Neural Architecture Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuralforge-nas --population 20 --generations 50
  neuralforge-nas --dataset cifar10 --population 15 --generations 30
        """
    )
    
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset for evaluation')
    parser.add_argument('--population', type=int, default=15, help='Population size')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    config = Config()
    config.device = args.device
    config.nas_enabled = True
    config.nas_population_size = args.population
    config.nas_generations = args.generations
    config.nas_mutation_rate = args.mutation_rate
    
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

if __name__ == '__main__':
    main()
