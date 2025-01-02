"""
Module defining the HeuristicInstance data structure for the FunSearch model.
Represents an individual heuristic with its function code.
"""

from pydantic import BaseModel, Field

class HeuristicInstance(BaseModel):
    """
    A simpler data structure for FunSearch that stores only a function code snippet.
    """
    function: str = Field(..., description="A full Python code snippet.")

    def __hash__(self):
        """
        Generates a unique hash based on the function.

        Returns:
            int: The hash value of the heuristic instance.
        """
        return hash(self.function)

    def __eq__(self, other):
        """
        Checks equality between two HeuristicInstance objects based on function.

        Args:
            other (HeuristicInstance): The other heuristic instance to compare.

        Returns:
            bool: True if functions are identical, False otherwise.
        """
        if not isinstance(other, HeuristicInstance):
            return False
        return self.function == other.function
