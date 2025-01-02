"""
Module defining prompt templates for the Evolution of Heuristics (EoH) model.
Includes templates for mutation and crossover operations to guide the language model.
"""

from langchain.prompts import PromptTemplate

# Prompt templates for performing mutations on heuristics
MUTATIONS_PROMPTS = [
    PromptTemplate(
        input_variables=["design", "function"],
        template="""
        I have one algorithm with its code as follows.
        ## Parent Function ##
        {{
            "design": {design}
            "function": {function}
        }}
        Modify this to create a new algorithm that has a different form but can be a modified version of the algorithm provided.
        First, describe your new algorithm and main steps with the "design" key, then implement it in Python with the "function" key.
        You MUST answer as a VALID json with the given format.
        """
    ),
    PromptTemplate(
        input_variables=["design", "function"],
        template="""
        I have one algorithm with its code as follows.
        ## Parent Function ##
        {{
            "design": {design}
            "function": {function}
        }}
        Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n\
        First, describe your new algorithm and main steps in one sentence. The description must be in the the "design" key, then implement it in Python with the "function" key.
        You MUST answer as a VALID json with the given format.
        """
    ),
    PromptTemplate(
        input_variables=["design", "function"],
        template="""
        I have one algorithm with its code as follows.
        ## Parent Function ##
        {{
            "design": {design}
            "function": {function}
        }}
        First, you need to identify the main components in the function above. Next, analyze whether any of these components can be overfit to the in-distribution instances. Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. Finally, provide the revised code.
        Give your results with the same "design" and "function" keys.
        You MUST answer as a VALID json with the given format.
        """
    )
]

# Prompt templates for performing crossover between two heuristics
CROSSOVER_PROMPTS = [
    PromptTemplate(
        input_variables=["parent1_design", "parent1_function", "parent2_design", "parent2_function"],
        template="""
        I have 2 existing algorithms with their codes as follows:
        ## Parent Function A ##
        {{
            "design": {parent1_design}
            "function": {parent1_function}
        }}

        ## Parent Function B ##:
        {{
            "design": {parent2_design}
            "function": {parent2_function}
        }}
        
        You must create a new algorithm that has a totally different form from the given ones.
        First, describe your new algorithm and main steps with the "design" key, then implement it in Python with the "function" key.
        You MUST answer as a VALID json with the given format.
        """
    ),
    PromptTemplate(
        input_variables=["parent1_design", "parent1_function", "parent2_design", "parent2_function"],
        template="""
        I have 2 existing algorithms with their codes as follows:
        ## Parent Function A ##
        {{
            "design": {parent1_design}
            "function": {parent1_function}
        }}

        ## Parent Function B ##:
        {{
            "design": {parent2_design}
            "function": {parent2_function}
        }}
        You MUST create a new function based on a common backbone idea from both parents.
        Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm and its steps with the "design" key, then implement it in Python with the "function" key.
        You MUST answer as a VALID json with the given format.
        """
    )
]
