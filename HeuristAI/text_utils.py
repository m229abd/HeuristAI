"""
Utility functions for handling and formatting code snippets.
Includes functions to clean and revert code formatting for consistent processing.
"""

from textwrap import dedent

def clean_code(text: str) -> str:
    """
    Cleans and formats the code by removing indentation and escaping newline and tab characters.

    Args:
        text (str): The raw code string.

    Returns:
        str: The cleaned and formatted code string.
    """
    return dedent(text).replace('\n', '\\n').replace('\t', '\\t')

def revert_code(text: str) -> str:
    """
    Reverts the cleaned code back to its original formatting by unescaping newline and tab characters.

    Args:
        text (str): The cleaned code string.

    Returns:
        str: The reverted code string with original formatting.
    """
    return text.replace('\\n', '\n').replace('\\t', '\t')
