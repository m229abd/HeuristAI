"""
Example script for running the Evolution of Heuristics (EoH) model.
This script initializes the EoH model, evaluates heuristic instances,
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
from examples.EvolutionOfHeuristics.search import SEARCH_SEED_INSTANCES, SEARCH_TEST_CASES
from models.EvolutionOfHeuristics import EvolutionOfHeuristics
from text_utils import revert_code

# Configure logging to display informational messages with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote
def default_fitness_function(instance, test_cases):
    """
    A generic fitness function example. 
    Adjust or replace this with your domain-specific evaluation. 
    It's easily passed to the EvolutionaryBaseModel constructor.
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

                # Simple scoring: 1/(1+time) for correct answers, else 0
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
    Main execution block for running the EoH example in an async context.
    Initializes Ray, sets up the language model, runs the evolutionary process,
    and outputs the best heuristic found.
    """
    ray.init()

    api_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize the ChatOpenAI language model
    gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # Instantiate and configure the EvolutionOfHeuristics model
    eoh = EvolutionOfHeuristics(
        population_size=4,
        generations=3,
        test_cases=SEARCH_TEST_CASES,
        llm=gpt_llm,
        fitness_function=default_fitness_function.remote,
        num_retries=3,
        log_level=logging.ERROR
    )
    eoh.initialize_population(SEARCH_SEED_INSTANCES)
    
    # Run the evolutionary process (async)
    best_eoh = await eoh.run()
    
    # Display the best heuristic found
    print("=== Best EoH instance ===")
    print("Design:", best_eoh.design)
    print("Function:", best_eoh.function)

    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
