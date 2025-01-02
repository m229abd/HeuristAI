"""
Example script for running the FunSearch evolutionary algorithm.
This script initializes the FunSearch model, evaluates heuristic instances across multiple islands,
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
from examples.FunSearch.search import SEARCH_SEED_INSTANCES, SEARCH_TEST_CASES
from models.FunSearch import FunSearch
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
    Main execution block for running the FunSearch example in an async context.
    """
    ray.init()

    api_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

    gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    
    funsearch = FunSearch(
        population_size=4,
        test_cases=SEARCH_TEST_CASES,
        llm=gpt_llm,
        num_islands=2,
        reset_interval=2,
        generations=5,
        k_parents=3,
        fitness_function=default_fitness_function.remote,
        num_retries=3
    )
    funsearch.initialize_islands(SEARCH_SEED_INSTANCES)
    
    # Run the FunSearch evolutionary process (async)
    best_fs = await funsearch.run()

    print("\n=== Best FunSearch program ===")
    print(best_fs.function)

    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
