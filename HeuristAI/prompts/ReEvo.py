"""
Module defining prompt templates for the ReEvo model.
Includes templates for short-term and long-term considerations, as well as elitist mutations.
"""

from langchain.prompts import PromptTemplate

# Prompt template for generating short-term considerations and offspring
SHORT_TERM_PROMPT = PromptTemplate(
    input_variables=["worse_code", "better_code"],
    template="""
Below is a "worse" code snippet and a "better" code snippet. 
Compare them, infer potential improvements, and produce JSON with "consideration" first and "function" second.
Use less than 50 words in the consideration.
You MUST answer just with a VALID json with the given format. 

Worse code:
{worse_code}

Better code:
{better_code}
    """
)

# Prompt template for merging long-term considerations
LONG_TERM_PROMPT = PromptTemplate(
    input_variables=["prior_considerations", "new_considerations"],
    template="""
Below is your prior long-term consideration history (concatenated), 
followed by newly gained short-term considerations. 
Merge them into a concise final consideration under 50 words. 
Return JSON with ONLY the "consideration" key first and then a empty "function" key.
You MUST answer just with a VALID json with the given format. 

Prior considerations:
{prior_considerations}

New considerations:
{new_considerations}
    """
)

# Prompt template for applying elitist mutations based on long-term considerations
ELITIST_MUTATION_PROMPT = PromptTemplate(
    input_variables=["long_term_consideration", "elite_code"],
    template="""
Below is a long-term consideration and the best code from the population. 
Mutate/improve the code according to the consideration. 
Produce JSON with "consideration" first (under 50 words) and "function" second.
You MUST answer just with a VALID json with the given format. 

Long-term consideration:
{long_term_consideration}

Elite code:
{elite_code}
    """
)
