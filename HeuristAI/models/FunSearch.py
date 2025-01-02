import logging
import random
import ray
import asyncio
from typing import List, Dict, Optional
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from text_utils import clean_code

from models.EvolutionaryBaseModel import EvolutionaryBaseModel
from structures.FunSearch import HeuristicInstance
from prompts.FunSearch import CROSSOVER_PROMPT
from utils import invoke_with_retries

logger = logging.getLogger(__name__)

class FunSearch(EvolutionaryBaseModel):
    """
    FunSearch implements an island-model evolutionary search with multi-parent crossover
    and periodic resets to maintain diversity within the population.
    """

    def __init__(
        self,
        population_size: int,
        test_cases: List[Dict],
        llm,
        num_islands: int = 2,
        reset_interval: int = 5,
        generations: int = 10,
        k_parents: int = 2,
        fitness_function=None,
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
        self.num_islands = num_islands
        self.reset_interval = reset_interval
        self.generations = generations
        self.k_parents = k_parents

        self.islands = [[] for _ in range(num_islands)]
        self.parser = PydanticOutputParser(pydantic_object=HeuristicInstance)

    def initialize_islands(self, seed_programs: List[HeuristicInstance]):
        for i in range(self.num_islands):
            self.islands[i] = list(seed_programs)
        self.evaluate_all_islands()

    def evaluate_all_islands(self):
        combined = []
        for i, island_pop in enumerate(self.islands):
            for ind in island_pop:
                combined.append((ind, i))

        if not combined:
            return

        tasks = [self.fitness_function(ind, self.test_cases) for (ind, _) in combined]
        results = ray.get(tasks)

        for ((ind, _), fit) in zip(combined, results):
            self.fitness_cache[ind] = fit

    def selection_in_island(self, island_idx: int, num_parents: int):
        island_pop = self.islands[island_idx]
        if len(island_pop) < num_parents:
            return random.sample(island_pop, k=len(island_pop))
        return self.selection(island_pop, num_parents)

    async def llm_crossover(self, parents: List[HeuristicInstance]) -> HeuristicInstance:
        parent_snippets = []
        for idx, p in enumerate(parents):
            parent_snippets.append("{\n\"function\" : \"" + clean_code(p.function) + "\"\n}")
        combined_parents = "\n".join(parent_snippets)

        chain = LLMChain(llm=self.llm, prompt=CROSSOVER_PROMPT)

        try:
            new_candidate = await invoke_with_retries(
                chain=chain,
                prompt_params={"parent_functions": combined_parents},
                parser=self.parser,
                max_retries=self.num_retries
            )
            return new_candidate
        except Exception as e:
            self.logger.error(f"[FunSearch] Crossover failed after {self.num_retries} retries: {e}")
            return None

    def island_reset(self):
        island_best_fitness = []
        for i in range(self.num_islands):
            if not self.islands[i]:
                island_best_fitness.append((i, float('-inf')))
                continue
            best_in_island = max(self.islands[i], key=lambda x: self.fitness_cache.get(x, 0))
            island_best_fitness.append((i, self.fitness_cache.get(best_in_island, 0)))

        island_best_fitness.sort(key=lambda x: x[1], reverse=True)
        half = self.num_islands // 2
        top_islands = island_best_fitness[:half]
        bottom_islands = island_best_fitness[half:]

        if not top_islands:
            self.logger.error("[FunSearch] All islands are empty or have zero fitness.")
            return

        top_island_index, _ = random.choice(top_islands)
        best_in_top_island = max(
            self.islands[top_island_index],
            key=lambda x: self.fitness_cache.get(x, 0)
        )

        for (bad_island_idx, _) in bottom_islands:
            self.islands[bad_island_idx] = [best_in_top_island]

    async def run(self):
        """
        Executes the main FunSearch loop, performing selection, crossover, evaluation,
        pruning, and periodic resetting across all islands.
        """
        for gen in range(self.generations):
            self.logger.info("[FunSearch] Generation %d / %d", gen+1, self.generations)

            crossover_coros = []
            new_candidates = []

            # Initiate crossover for each island concurrently
            for island_idx in range(self.num_islands):
                parents = self.selection_in_island(island_idx, self.k_parents)
                if len(parents) < self.k_parents:
                    self.logger.warning(f"[FunSearch] Not enough parents in island {island_idx} for crossover.")
                    continue
                coro = self.llm_crossover(parents)
                crossover_coros.append((island_idx, coro))

            # Execute all crossover operations concurrently
            if crossover_coros:
                results = await asyncio.gather(*[coro for (_, coro) in crossover_coros], return_exceptions=True)
                for idx, (island_idx, _) in enumerate(crossover_coros):
                    result = results[idx]
                    if isinstance(result, Exception):
                        self.logger.error(f"[FunSearch] Crossover in island {island_idx} failed: {result}")
                        continue
                    if result:
                        self.islands[island_idx].append(result)
                        new_candidates.append(result)

            if new_candidates:
                self.evaluate_all_islands()
                # Prune populations after adding new candidates
                for island_idx in range(self.num_islands):
                    self.islands[island_idx] = self.prune_population(self.islands[island_idx])

            # Periodic island reset
            if (gen + 1) % self.reset_interval == 0 and gen < self.generations - 1:
                self.island_reset()
                self.evaluate_all_islands()

        # Combine final results
        all_individuals = []
        for i in range(self.num_islands):
            all_individuals.extend(self.islands[i])

        if not all_individuals:
            self.logger.error("[FunSearch] No individuals at the end of run.")
            return None

        best = max(all_individuals, key=lambda x: self.fitness_cache.get(x, 0))
        self.logger.info("[FunSearch] Best program: %s", best.function)
        return best
