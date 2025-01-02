import logging
import random
import ray
import json
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class EvolutionaryBaseModel:
    """
    A base class providing common evolutionary functionalities for derived models.
    Includes methods for evaluating fitness, selecting parents, pruning the population,
    and logging the population and fitness to a JSON file.
    """

    def __init__(
        self,
        population_size: int,
        test_cases: List[Dict],
        llm,
        fitness_function,
        num_retries: int = 3,
        log_file: Optional[str] = "evolution_log.json",
        log_level: int = logging.INFO
    ):
        """
        Initializes the EvolutionaryBaseModel with the specified parameters.

        Args:
            population_size (int): Maximum number of individuals in the population.
            test_cases (list): List of test cases for evaluating fitness.
            llm: Language model instance for generating code.
            fitness_function: Ray-remote function to evaluate fitness of individuals.
            num_retries (int): Number of retry attempts for LLM calls.
            log_file (str): Path to the JSON file for logging population and fitness.
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self.population_size = population_size
        self.population = []
        self.test_cases = test_cases
        self.fitness_function = fitness_function
        self.fitness_cache = {}
        self.llm = llm
        self.num_retries = num_retries

        # Initialize logging
        self.log_file_path = Path(log_file)
        self.initialize_logging(log_level)

    def initialize_logging(self, log_level: int):
        """
        Initializes the logging configuration and the JSON log file.

        Args:
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        # Configure the root logger only once
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        # Set the logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.log_file_path.exists():
            with self.log_file_path.open("w") as f:
                json.dump({"logs": []}, f, indent=4)
            self.logger.info(f"[BaseModel] Created new log file at {self.log_file_path}")
        else:
            self.logger.info(f"[BaseModel] Using existing log file at {self.log_file_path}")

    def log_population(self, generation: int):
        """
        Logs the current population and their fitness scores to the JSON log file.

        Args:
            generation (int): The current generation or iteration number.
        """
        log_entry = {
            "generation": generation,
            "individuals": []
        }

        for individual in self.population:
            # Assuming HeuristicInstance or similar with .dict() method
            individual_data = individual.dict() if hasattr(individual, "dict") else {"function": str(individual)}
            fitness = self.fitness_cache.get(individual, 0)
            individual_data["fitness"] = fitness
            log_entry["individuals"].append(individual_data)

        # Read existing logs
        try:
            with self.log_file_path.open("r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"[BaseModel] JSON decode error: {e}. Initializing new log structure.")
            data = {"logs": []}

        # Append the new log entry
        data["logs"].append(log_entry)

        # Write back to the log file
        try:
            with self.log_file_path.open("w") as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"[BaseModel] Logged generation {generation} to {self.log_file_path}")
        except Exception as e:
            self.logger.error(f"[BaseModel] Failed to write to log file: {e}")

    def evaluate_population(self, population: List):
        """
        Evaluates the fitness of the entire population in parallel using Ray.
        Logs the population and their fitness scores.

        Args:
            population (list): List of individuals to evaluate.

        Returns:
            dict: A dictionary mapping individuals to their fitness scores.
        """
        if not population:
            self.logger.warning("[BaseModel] Empty population received for evaluation.")
            return {}

        # Prepare fitness evaluation tasks
        fitness_tasks = [
            self.fitness_function(ind, self.test_cases) for ind in population
        ]
        self.logger.info("[BaseModel] Evaluating population fitness...")
        # Retrieve fitness scores from Ray
        fitness_scores = ray.get(fitness_tasks)
        cache = {individual: score for individual, score in zip(population, fitness_scores)}
        self.fitness_cache = cache

        # Determine current generation number based on existing logs
        try:
            with self.log_file_path.open("r") as f:
                data = json.load(f)
                current_generation = len(data.get("logs", [])) + 1
        except json.JSONDecodeError:
            self.logger.error("[BaseModel] JSON decode error while determining generation number.")
            current_generation = 1

        # Log the current population and fitness
        self.log_population(generation=current_generation)

        return cache

    def selection(self, population: List, num_parents: int) -> List:
        """
        Selects parents for reproduction using fitness-proportional selection without replacement.

        Args:
            population (list): List of individuals in the population.
            num_parents (int): Number of parents to select.

        Returns:
            list: List of selected parent individuals sorted by fitness in descending order.
        """
        total_fitness = sum(self.fitness_cache.get(ind, 0) for ind in population)
        
        if total_fitness == 0:
            self.logger.warning("[BaseModel] Total fitness is zero. Selecting randomly.")
            selected = random.sample(population, k=num_parents)
            return selected
        
        population_copy = population.copy()
        fitness_dict = {ind: self.fitness_cache.get(ind, 0) for ind in population_copy}
        
        selected = []
        
        for _ in range(num_parents):
            current_total_fitness = sum(fitness_dict[ind] for ind in population_copy)
            
            if current_total_fitness == 0:
                selected.extend(random.sample(population_copy, k=num_parents - len(selected)))
                break
            
            probabilities = [
                fitness_dict[ind] / current_total_fitness for ind in population_copy
            ]
            
            chosen = random.choices(population_copy, weights=probabilities, k=1)[0]
            selected.append(chosen)
            
            population_copy.remove(chosen)
        
        # Sort the selected parents by fitness in descending order
        selected_sorted = sorted(selected, key=lambda ind: self.fitness_cache.get(ind, 0), reverse=True)
        
        return selected_sorted

    def prune_population(self, population: List) -> List:
        """
        Prunes the population to retain only the top-performing individuals based on fitness.

        Args:
            population (list): Current population of individuals.

        Returns:
            list: Pruned population containing only the top individuals.
        """
        if len(population) <= self.population_size:
            self.logger.info("[BaseModel] Population size within limit. No pruning needed.")
            return population

        # Sort population by fitness in descending order and retain the top individuals
        sorted_pop = sorted(
            population,
            key=lambda x: self.fitness_cache.get(x, 0),
            reverse=True
        )
        pruned_population = sorted_pop[:self.population_size]
        self.logger.info("[BaseModel] Pruned population to top %d individuals.", self.population_size)
        return pruned_population
