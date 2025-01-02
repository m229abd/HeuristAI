"""
Module defining the HeuristicInstance data structure for the ReEvo model.
Represents an individual heuristic with its consideration and associated function code.
"""

from pydantic import BaseModel, Field

class HeuristicInstance(BaseModel):
    """
    A data structure to hold the results of ReEvo steps,
    containing short or long-term consideration and a function.
    """
    consideration: str = Field(..., description="Short or long-term consideration text.")
    function: str = Field(..., description="A Python function code snippet.")

    def __hash__(self):
        """
        Generates a unique hash based on the consideration and function.

        Returns:
            int: The hash value of the heuristic instance.
        """
        return hash((self.consideration, self.function))

    def __eq__(self, other):
        """
        Checks equality between two HeuristicInstance objects based on consideration and function.

        Args:
            other (HeuristicInstance): The other heuristic instance to compare.

        Returns:
            bool: True if both consideration and function are identical, False otherwise.
        """
        if not isinstance(other, HeuristicInstance):
            return False
        return self.consideration == other.consideration and self.function == other.function
