import logging
import asyncio
from typing import List, Optional
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from models.EvolutionaryBaseModel import EvolutionaryBaseModel
from prompts.ReEvo import (
    SHORT_TERM_PROMPT,
    LONG_TERM_PROMPT,
    ELITIST_MUTATION_PROMPT
)
from structures.ReEvo import HeuristicInstance
from utils import invoke_with_retries

logger = logging.getLogger(__name__)

class ReEvo(EvolutionaryBaseModel):
    """
    ReEvo implements an evolutionary algorithm with mechanisms for handling
    short-term and long-term considerations, elitist mutations, and population management.
    """

    def __init__(
        self,
        population_size: int,
        max_iterations: int,
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
        self.max_iterations = max_iterations

        self.population: List[HeuristicInstance] = []
        self.long_term_considerations = []

        self.parser = PydanticOutputParser(pydantic_object=HeuristicInstance)

    def initialize_population(self, seed_heuristics: List[HeuristicInstance]):
        self.population = list(seed_heuristics)
        self.fitness_cache = self.evaluate_population(self.population)

    def parse_heuristic_output(self, text_output: str) -> HeuristicInstance:
        try:
            new_heur = self.parser.parse(text_output)
            return new_heur
        except ValidationError as e:
            self.logger.error("[ReEvo] Could not parse LLM output as ReEvoHeuristic: %s", e)
            return HeuristicInstance(
                consideration="",
                function=text_output
            )

    async def run(self):
        """
        Executes the main ReEvo loop, performing selection, generating offspring,
        handling considerations, applying elitist mutations, and pruning the population.
        """
        iteration_count = 0

        while iteration_count < self.max_iterations:
            self.logger.info("[ReEvo] Iteration %d / %d", iteration_count+1, self.max_iterations)

            # Select parent pairs
            parent_pairs = []
            num_pairs = len(self.population) // 2
            for _ in range(num_pairs):
                if len(self.population) < 2:
                    break
                parents = self.selection(self.population, 2)
                if len(parents) == 2:
                    parent_pairs.append((parents[0], parents[1]))

            short_term_considerations = []
            offspring_list = []

            # Generate offspring and short-term considerations (async)
            async_coros = []
            for (better, worse) in parent_pairs:
                chain = LLMChain(llm=self.llm, prompt=SHORT_TERM_PROMPT)
                # Using invoke_with_retries
                coro = invoke_with_retries(
                    chain=chain,
                    prompt_params={
                        "worse_code": worse.function,
                        "better_code": better.function
                    },
                    parser=self.parser,
                    max_retries=self.num_retries
                )
                async_coros.append(coro)

            try:
                raw_outputs = await asyncio.gather(*async_coros)
            except Exception as e:
                self.logger.error(f"[ReEvo] Failed to generate offspring after retries: {e}")
                continue  # Skip this iteration or handle as needed

            # Parse each output
            for new_offspring in raw_outputs:
                short_term_considerations.append(new_offspring.consideration)
                offspring_list.append(new_offspring)

            # Merge short-term considerations into long-term
            if short_term_considerations:
                joined_prior = "\n".join(self.long_term_considerations)
                joined_new = "\n".join(short_term_considerations)
                chain_lt = LLMChain(llm=self.llm, prompt=LONG_TERM_PROMPT)
                try:
                    lt_output = await invoke_with_retries(
                        chain=chain_lt,
                        prompt_params={
                            "prior_considerations": joined_prior,
                            "new_considerations": joined_new
                        },
                        parser=self.parser,
                        max_retries=self.num_retries
                    )
                    long_term_result = lt_output
                    self.long_term_considerations.append(long_term_result.consideration)
                except Exception as e:
                    self.logger.error(f"[ReEvo] Failed to merge considerations after retries: {e}")

            # Elitist mutation
            best_heur = self.get_best_heuristic(self.population)
            ltc = self.long_term_considerations[-1] if self.long_term_considerations else ""
            chain_elite = LLMChain(llm=self.llm, prompt=ELITIST_MUTATION_PROMPT)
            try:
                raw_elite = await invoke_with_retries(
                    chain=chain_elite,
                    prompt_params={
                        "long_term_consideration": ltc,
                        "elite_code": best_heur.function
                    },
                    parser=self.parser,
                    max_retries=self.num_retries
                )
                mutated_heur = raw_elite
                # Update population
                self.population.append(mutated_heur)
            except Exception as e:
                self.logger.error(f"[ReEvo] Failed to perform elitist mutation after retries: {e}")

            # Update population with offspring
            self.population.extend(offspring_list)

            # Evaluate population via Ray
            self.fitness_cache = self.evaluate_population(self.population)
            self.population = self.prune_population(self.population)

            iteration_count += 1

        return self.get_best_heuristic(self.population)

    def get_best_heuristic(self, population: List[HeuristicInstance]) -> HeuristicInstance:
        if not population:
            raise ValueError("[ReEvo] Population is empty; no best heuristic.")
        return max(population, key=lambda h: self.fitness_cache.get(h, 0))
