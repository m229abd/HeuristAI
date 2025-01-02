"""
Module defining prompt templates for the FunSearch model.
Includes templates for multi-parent crossover operations.
"""

from langchain.prompts import PromptTemplate

# Prompt template for performing multi-parent crossover to generate a new heuristic
CROSSOVER_PROMPT = PromptTemplate(
    input_variables=["parent_functions"],
    template="""
{parent_functions}

Generate a combined design and function (just answer as a VALID json with "function" key):
"""
)
