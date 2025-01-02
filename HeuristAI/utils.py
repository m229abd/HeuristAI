import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

async def invoke_with_retries(chain, prompt_params: Dict[str, Any], parser, max_retries: int):
    """
    Invokes an LLMChain with retries upon failure.

    Args:
        chain: The LLMChain instance to invoke.
        prompt_params (dict): Parameters to pass to the prompt.
        parser: The PydanticOutputParser to parse the LLM's output.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        Parsed object from the LLM's output.

    Raises:
        Exception if all retries fail.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            response = await chain.ainvoke(prompt_params)
            parsed_output = parser.parse(response['text'])
            return parsed_output
        except Exception as e:
            logger.warning(f"LLM invocation failed on attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"All {max_retries} attempts failed.")
                raise e
            await asyncio.sleep(1)