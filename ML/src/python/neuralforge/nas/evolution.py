import torch
import random
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from .search_space import SearchSpace, Architecture
from .evaluator import ModelEvaluator

class EvolutionarySearch:
    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: ModelEvaluator,
        population_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        tournament_size: int = 3
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        self.population = []
        self.best_architecture = None
        self.history = []
    
    def initialize_population(self):
        print(f"Initializing population of {self.population_size} architectures...")
        self.population = []
        
        for i in range(self.population_size):
            arch = self.search_space.random_architecture()
            self.population.append(arch)
        
        print("Population initialized successfully")
    
    def evaluate_population(self):
        print("Evaluating population...")
        
        for arch in tqdm(self.population, desc="Evaluating architectures"):
            if arch.fitness == 0.0:
                fitness, accuracy = self.evaluator.evaluate(arch, self.search_space)
                arch.fitness = fitness
                arch.accuracy = accuracy
                
                complexity = self.search_space.estimate_complexity(arch)
                arch.params = complexity['params']
                arch.flops = complexity['flops']
    
    def tournament_selection(self) -> Architecture:
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def select_parents(self) -> List[Architecture]:
        parent1 = self.tournament_selection()
        parent2 = self.tournament_selection()
        return [parent1, parent2]
    
    def create_offspring(self, parents: List[Architecture]) -> Architecture:
        if random.random() < self.crossover_rate:
            offspring = self.search_space.crossover(parents[0], parents[1])
        else:
            offspring = Architecture(parents[0].genome.copy())
        
        if random.random() < self.mutation_rate:
            offspring = self.search_space.mutate(offspring, self.mutation_rate)
        
        return offspring
    
    def evolve_generation(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        elite_size = max(1, self.population_size // 10)
        new_population = self.population[:elite_size]
        
        while len(new_population) < self.population_size:
            parents = self.select_parents()
            offspring = self.create_offspring(parents)
            new_population.append(offspring)
        
        self.population = new_population
    
    def search(self) -> Architecture:
        print(f"Starting evolutionary search for {self.generations} generations...")
        
        self.initialize_population()
        self.evaluate_population()
        
        for generation in range(self.generations):
            print(f"\n=== Generation {generation + 1}/{self.generations} ===")
            
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_arch = self.population[0]
            
            if self.best_architecture is None or best_arch.fitness > self.best_architecture.fitness:
                self.best_architecture = best_arch
            
            avg_fitness = np.mean([arch.fitness for arch in self.population])
            avg_accuracy = np.mean([arch.accuracy for arch in self.population])
            
            print(f"Best fitness: {best_arch.fitness:.4f}")
            print(f"Best accuracy: {best_arch.accuracy:.2f}%")
            print(f"Avg fitness: {avg_fitness:.4f}")
            print(f"Avg accuracy: {avg_accuracy:.2f}%")
            print(f"Best params: {best_arch.params:,}")
            
            self.history.append({
                'generation': generation + 1,
                'best_fitness': best_arch.fitness,
                'best_accuracy': best_arch.accuracy,
                'avg_fitness': avg_fitness,
                'avg_accuracy': avg_accuracy,
            })
            
            if generation < self.generations - 1:
                self.evolve_generation()
                self.evaluate_population()
        
        print(f"\nSearch completed! Best architecture: {self.best_architecture}")
        return self.best_architecture
    
    def get_top_k_architectures(self, k: int = 5) -> List[Architecture]:
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[:k]