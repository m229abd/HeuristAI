"""
Example script for running the ReEvo evolutionary algorithm.
This script initializes the ReEvo model, evaluates heuristic instances,
and identifies the best-performing heuristic based on predefined test cases.
"""

import os
import logging
import re
import time
import io
import sys
import ray
import asyncio

from langchain_openai import ChatOpenAI
from models.ReEvo import ReEvo
from examples.ReEvo.search import SEARCH_SEED_INSTANCES, SEARCH_TEST_CASES
from text_utils import revert_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote
def default_fitness_function(instance, test_cases):
    """
    Evaluates the fitness of a heuristic instance by executing its function
    against a set of test cases and scoring based on correctness and execution time.
    """
    code = revert_code(instance.function)

    try:
        function_name_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not function_name_match:
            logger.error("[FitnessFunc] No function definition found in code snippet.")
            return 0
        function_name = function_name_match.group(1)

        fitness_score = 0
        for test in test_cases:
            test_case = test['test_case']
            expected_answer = test['answer']

            local_vars = {}
            output_capture = io.StringIO()
            sys.stdout = output_capture
            try:
                exec(code, {}, local_vars)
                if function_name not in local_vars:
                    logger.error("[FitnessFunc] Function %s not defined after exec.", function_name)
                    return 0

                start_time = time.time()
                result = local_vars[function_name](test_case)
                end_time = time.time()
                execution_time = end_time - start_time

                printed_output = output_capture.getvalue()
                logger.debug("Execution time for %s: %f seconds", function_name, execution_time)
                logger.debug("Output: %s", printed_output)

                if result == expected_answer:
                    fitness_score += 1 / (1 + execution_time)
            except Exception as e:
                logger.error("[FitnessFunc] Error during execution of %s: %s", function_name, e)
            finally:
                sys.stdout = sys.__stdout__

        return fitness_score
    except Exception as e:
        logger.error("[FitnessFunc] Error evaluating fitness: %s", e)
        return 0

async def main():
    """
    Main execution block for running the ReEvo example in an async context.
    """
    ray.init()

    api_key = input("OpenAI key: ")
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    re_evo = ReEvo(
        population_size=5,
        max_iterations=3,
        test_cases=SEARCH_TEST_CASES,
        llm=llm,
        fitness_function=default_fitness_function.remote,
        num_retries=3
    )

    re_evo.initialize_population(SEARCH_SEED_INSTANCES)

    # Run the ReEvo evolutionary process (async)
    best_heuristic = await re_evo.run()

    print("\n[Final best ReEvoHeuristic]")
    print("Consideration:", best_heuristic.consideration)
    print("Function:\n", best_heuristic.function)

    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
