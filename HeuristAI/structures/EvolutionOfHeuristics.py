"""
Module defining the HeuristicInstance data structure for the EvolutionOfHeuristics model.
Represents an individual heuristic with its design and associated function code.
"""

from pydantic import BaseModel, Field

class HeuristicInstance(BaseModel):
    """
    A data structure to represent a heuristic instance in the EvolutionOfHeuristics model.
    Contains both the design description and the corresponding function implementation.
    """
    design: str = Field(..., description="The design or heuristic associated with the function.")
    function: str = Field(..., description="The Python function code snippet.")

    def __hash__(self):
        """
        Generates a unique hash based on the design and function.

        Returns:
            int: The hash value of the heuristic instance.
        """
        return hash((self.design, self.function))

    def __eq__(self, other):
        """
        Checks equality between two HeuristicInstance objects based on design and function.

        Args:
            other (HeuristicInstance): The other heuristic instance to compare.

        Returns:
            bool: True if both design and function are identical, False otherwise.
        """
        if not isinstance(other, HeuristicInstance):
            return False
        return self.design == other.design and self.function == other.function
