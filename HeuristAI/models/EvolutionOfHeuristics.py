import logging
from typing import Optional
import asyncio
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from models.EvolutionaryBaseModel import EvolutionaryBaseModel
from prompts.EvolutionOfHeuristics import MUTATIONS_PROMPTS, CROSSOVER_PROMPTS
from structures.EvolutionOfHeuristics import HeuristicInstance
from utils import invoke_with_retries

logger = logging.getLogger(__name__)

class EvolutionOfHeuristics(EvolutionaryBaseModel):
    """
    EvolutionOfHeuristics implements an evolutionary algorithm to evolve
    heuristic instances by performing crossover and mutation operations.
    """

    def __init__(
        self,
        population_size,
        generations,
        test_cases,
        llm,
        fitness_function,
        num_retries: int = 3,
        log_file: Optional[str] = "evolution_log.json",
        log_level: int = logging.INFO
    ):
        super().__init__(
            population_size=population_size,
            test_cases=test_cases,
            llm=llm,
            fitness_function=fitness_function,
            num_retries=num_retries,
            log_file=log_file,
            log_level=log_level
        )
        self.generations = generations
        self.parser = PydanticOutputParser(pydantic_object=HeuristicInstance)

    def initialize_population(self, seed_instances):
        """
        Initializes the population with seed heuristic instances and evaluates their fitness.
        """
        self.population = list(seed_instances)
        self.fitness_cache = self.evaluate_population(self.population)

    def parse_to_heuristic_instance(self, generated_code):
        """
        Parses generated code into a HeuristicInstance object.
        """
        try:
            instance = self.parser.parse(generated_code.replace('\n', ''))
            return instance
        except ValidationError as e:
            self.logger.error("[EoH] Could not parse LLM output as HeuristicInstance: %s", e)
            return None
    
    async def crossover_operation(self, parent1, parent2, crossover_prompt):
        """
        Performs a crossover operation between two parent heuristics using a specific prompt to produce a child heuristic.
        Incorporates retry logic.
        """
        chain = LLMChain(llm=self.llm, prompt=crossover_prompt)
        try:
            child = await invoke_with_retries(
                chain=chain,
                prompt_params={
                    "parent1_design": parent1.design,
                    "parent1_function": parent1.function,
                    "parent2_design": parent2.design,
                    "parent2_function": parent2.function
                },
                parser=self.parser,
                max_retries=self.num_retries
            )
            return child
        except Exception as e:
            self.logger.error(f"[EoH] Crossover operation failed after {self.num_retries} retries: {e}")
            return None

    
    async def mutation_operation(self, candidate, mutation_prompt):
        """
        Performs a mutation operation on a candidate heuristic using a specific prompt to produce a mutated heuristic.
        Incorporates retry logic.
        """
        chain = LLMChain(llm=self.llm, prompt=mutation_prompt)
        try:
            mutant = await invoke_with_retries(
                chain=chain,
                prompt_params={
                    "design": candidate.design,
                    "function": candidate.function
                },
                parser=self.parser,
                max_retries=self.num_retries
            )
            return mutant
        except Exception as e:
            self.logger.error(f"[EoH] Mutation operation failed after {self.num_retries} retries: {e}")
            return None
        
        
    async def run(self):
        """
        Executes the main evolutionary loop, performing selection, crossover,
        mutation, and pruning over the specified number of generations.
        Ensures every crossover and mutation operation is applied at least once per generation.
        """
        for generation in range(self.generations):
            self.logger.info("[EoH] Generation %d / %d", generation + 1, self.generations)

            # Ensure every crossover prompt is applied once
            crossover_coros = []
            for crossover_prompt in CROSSOVER_PROMPTS:
                parents = self.selection(self.population, 2)
                if len(parents) < 2:
                    self.logger.warning("[EoH] Not enough parents for crossover. Skipping.")
                    continue
                coro = self.crossover_operation(parents[0], parents[1], crossover_prompt)
                crossover_coros.append(coro)

            # Ensure every mutation prompt is applied once
            mutation_coros = []
            for mutation_prompt in MUTATIONS_PROMPTS:
                candidate = self.selection(self.population, 1)[0]
                coro = self.mutation_operation(candidate, mutation_prompt)
                mutation_coros.append(coro)

            # Run all crossover and mutation operations concurrently
            all_coros = crossover_coros + mutation_coros
            try:
                results = await asyncio.gather(*all_coros)
            except Exception as e:
                self.logger.error(f"[EoH] Error during crossover/mutation operations: {e}")
                continue  # Proceed to next generation or handle as needed

            # Filter out None results
            new_offspring = [res for res in results if res is not None]

            # Add new offspring to the population
            self.population.extend(new_offspring)

            # Evaluate the updated population
            self.fitness_cache = self.evaluate_population(self.population)

            # Prune the population to maintain the population size
            self.population = self.prune_population(self.population)

            self.logger.info("[EoH] Generation %d completed. Population size: %d", generation + 1, len(self.population))

        # Identify and return the best heuristic
        if not self.population:
            self.logger.error("[EoH] Population is empty after evolution.")
            return None

        best_individual = max(self.population, key=lambda x: self.fitness_cache.get(x, 0))
        self.logger.info("[EoH] Best solution found: %s", best_individual)
        return best_individual
