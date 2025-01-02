# HeuristAI (v0.1.0)

## Overview

**HeuristAI** is a Python package that demonstrates how Large Language Models (LLMs) can enhance **Genetic Programming (GP)**. Traditional GP evolves programs or heuristics by applying random mutations and crossover operations, often termed “blind” due to their lack of external knowledge beyond the fitness function. By integrating LLMs (e.g., GPT-4-based models), **HeuristAI** enables more informed and intelligent manipulations of candidate solutions. Leveraging the vast **domain knowledge** of LLMs, the package facilitates the generation of refined variations rather than purely random ones, significantly improving the efficiency and effectiveness of the evolutionary search process.

---

## Table of Contents

1. [Overview](#overview)
2. [Citations](#citations)
3. [Why Genetic Programming with LLMs?](#why-genetic-programming-with-llms)
4. [Features](#features)
5. [Implemented Models](#implemented-models)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Usage](#usage)
    - [1. FunSearch](#1-funsearch)
    - [2. ReEvo](#2-reevo)
    - [3. Evolution of Heuristics (EoH)](#3-evolution-of-heuristics-eoh)
9. [Parallelization & Concurrency](#parallelization--concurrency)
10. [Customization](#customization)
11. [License](#license)

---

## Citations

HeuristAI includes implementations based on the following key research papers:

1. **DeepMind FunSearch** (*Nature*)  
   **Citation:**  
   Romera-Paredes, Bernardino, Barekatain, Mohammadamin, Novikov, Alexander, Balog, Matej, Kumar, M. Pawan, Dupont, Emilien, Ruiz, Francisco J. R., Ellenberg, Jordan S., Wang, Pengming, Fawzi, Omar, Kohli, Pushmeet, and Fawzi, Alhussein. *Mathematical discoveries from program search with large language models.* **Nature**, 625(7995): 468-475, 2024.  
   **DOI:** [10.1038/s41586-023-06924-6](https://doi.org/10.1038/s41586-023-06924-6)  
   **URL:** [Nature Article](http://dblp.uni-trier.de/db/journals/nature/nature625.html#RomeraParedesBNBKDREWFKF24)

2. **ReEvo** (*NeurIPS 2024*)  
   **Citation:**  
   Ye, Haoran, Wang, Jiarui, Cao, Zhiguang, Berto, Federico, Hua, Chuanbo, Kim, Haeyeon, Park, Jinkyoo, Song, Guojie. *ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution.* **arXiv preprint**, 2402.01145, 2024.  
   **URL:** [arXiv Paper](https://arxiv.org/abs/2402.01145)

3. **Evolution of Heuristics (EoH)** (*ICML 2024*)  
   **Citation:**  
   Liu, Fei, Tong, Xialiang, Yuan, Mingxuan, Lin, Xi, Luo, Fu, Wang, Zhenkun, Lu, Zhichao, Zhang, Qingfu. *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model.* **arXiv preprint**, 2401.02051, 2024.  
   **URL:** [arXiv Paper](https://arxiv.org/abs/2401.02051)

These implementations faithfully reproduce the methodologies and prompt structures as outlined in their respective publications.

---

## Why Genetic Programming with LLMs?

Traditional **Genetic Programming (GP)** evolves programs or heuristics by randomly mutating and combining parts of existing code, guided solely by a fitness function. While powerful, this approach can lead to inefficient explorations of the solution space.

**Incorporating Large Language Models (LLMs) offers several advantages:**

- **Informed Mutations and Crossover:** LLMs can suggest meaningful code modifications based on vast training data, ensuring that mutations and crossover operations are more likely to produce functional and efficient heuristics.
  
- **Domain Expertise:** LLMs possess extensive domain knowledge, allowing them to incorporate established algorithms and best practices into the evolutionary process, reducing the randomness inherent in traditional GP.
  
- **Reduced Blindness:** By leveraging LLMs, the evolutionary process becomes less "blind," focusing on more promising regions of the solution space and potentially discovering novel solutions that purely random methods might miss.

This synergy between GP and LLMs enhances the search capabilities, making the evolutionary process more efficient and effective.

---

## Features

1. **Easy to Use and Modify**  
   - Easily configure and modify models.
   - Define custom initial populations and fitness functions without altering core logic.

2. **Structured LLM Outputs**  
   - Ensures clean code generation with error-free parsing by enforcing structured responses from LLMs.

3. **Prompt Alignment with Research Papers**  
   - Utilizes prompt templates that closely mirror those used in foundational research, ensuring methodological consistency.

4. **Concurrent LLM Calls with LangChain**  
   - Leverages [LangChain](https://github.com/hwchase17/langchain) and `asyncio` for asynchronous and concurrent interactions with LLMs, speeding up the generation process.

5. **Parallel Fitness Computation with Ray**  
   - Utilizes [Ray](https://github.com/ray-project/ray) to parallelize fitness evaluations, effectively bypassing Python’s Global Interpreter Lock (GIL) for enhanced performance.

6. **Population Tracking**  
   - Logs each generation’s population and their fitness scores in a JSON file (`evolution_log.json`) for easy analysis and reproducibility.

7. **Model-Specific Enhancements**  
   - **FunSearch**: Transitioned from using only completion-based models (e.g., GPT-3.5) to instruction-based models (e.g., GPT-4) to align with API updates and deprecations.
   - **ReEvo**: Optimized token usage by generating reflections and code in a single LLM call, reducing the total token count by half.

8. **Ease of Using Different LLMs**  
   - **Flexible LLM Integration:** HeuristAI abstracts the LLM interface, allowing seamless integration of various language models such as GPT-4o, Llama-3 or other compatible models.
   - **Minimal Configuration Changes:** Switching between different LLMs requires minimal adjustments, enabling users to leverage the strengths of different models without extensive code modifications.
   - **Custom LLM Support:** Beyond OpenAI’s models, HeuristAI can integrate with other LLM providers by adhering to the standardized interface, facilitating broader applicability and customization.

---

## Implemented Models

HeuristAI currently includes implementations for the following models:

1. **FunSearch**  
   - **Description**: An island-model genetic programming approach that employs multi-parent crossover and periodic population resets to maintain diversity and avoid local maxima.
   - **Publication**: *Mathematical discoveries from program search with large language models.* [Nature](https://doi.org/10.1038/s41586-023-06924-6)
   - **Example Script**: [`examples/example_FunSearch.py`](./examples/example_FunSearch.py)

2. **ReEvo**  
   - **Description**: An evolutionary algorithm that utilizes short-term and long-term considerations to iteratively improve heuristic functions. Incorporates elitist mutations based on aggregated considerations.
   - **Publication**: *ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution.* [arXiv](https://arxiv.org/abs/2402.01145)
   - **Example Script**: [`examples/example_ReEvo.py`](./examples/example_ReEvo.py)

3. **Evolution of Heuristics (EoH)**  
   - **Description**: A standard genetic programming model enhanced with LLM-guided mutation and crossover operations. Utilizes prompt templates to steer the generation of new heuristics.
   - **Publication**: *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model.* [arXiv](https://arxiv.org/abs/2401.02051)
   - **Example Script**: [`examples/example_EvolutionofHeuristics.py`](./examples/example_EvolutionofHeuristics.py)

Check the `examples/` directory for detailed usage demonstrations of each model.

---

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/m229abd/HeuristAI.git
    cd HeuristAI
    ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the Package and Dependencies**

    You can install using `pip`:

    ```bash
    pip install -e .
    ```

    The `-e` flag installs the package in editable mode, allowing you to make changes to the source code that are immediately reflected without reinstalling.

    Alternatively, install dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**

    Ensure you have your OpenAI API key set up. You can set it as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
    ```

    Or input it when prompted by the example scripts.

---

## Quick Start

1. **Obtain an OpenAI API Key**

    Register and obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys).

2. **Run an Example Script**

    Navigate to the `examples/` directory and run one of the example scripts. For instance, to run **ReEvo**:

    ```bash
    cd examples
    python example_ReEvo.py
    ```

    You will be prompted to enter your OpenAI API key if it's not set as an environment variable.

3. **Monitor Evolution**

    The evolutionary process logs each generation’s population and fitness scores to `evolution_log.json`. You can monitor this file to track progress and analyze results.

---

## Usage

### 1. FunSearch

**FunSearch** is an island-model evolutionary algorithm that maintains multiple sub-populations (islands) evolving independently. Periodically, the best individuals from the strongest islands replace the worst-performing islands to maintain diversity and drive the search towards optimal solutions.

**Example:**

[`examples/example_FunSearch.py`](./examples/example_FunSearch.py)

```python
import asyncio
from HeuristAI.examples.example_FunSearch import main

if __name__ == "__main__":
    asyncio.run(main())
```

**Running the Example:**

```bash
python examples/example_FunSearch.py
```

### 2. ReEvo

**ReEvo** leverages short-term and long-term considerations to iteratively improve heuristic functions. It compares "worse" code snippets to "better" ones, generates improved offspring, aggregates considerations, and applies elitist mutations based on these aggregated insights.

**Example:**

[`examples/example_ReEvo.py`](./examples/example_ReEvo.py)

```python
import asyncio
from HeuristAI.examples.example_ReEvo import main

if __name__ == "__main__":
    asyncio.run(main())
```

**Running the Example:**

```bash
python examples/example_ReEvo.py
```

### 3. Evolution of Heuristics (EoH)

**Evolution of Heuristics (EoH)** uses standard evolutionary operators—mutation and crossover—enhanced by LLM-guided prompts to generate and refine heuristics. It ensures every crossover and mutation operation is applied at least once per generation, maintaining a robust search process.

**Example:**

[`examples/example_EvolutionofHeuristics.py`](./examples/example_EvolutionofHeuristics.py)

```python
import asyncio
from HeuristAI.examples.example_EvolutionofHeuristics import main

if __name__ == "__main__":
    asyncio.run(main())
```

**Running the Example:**

```bash
python examples/example_EvolutionofHeuristics.py
```

---

## Parallelization & Concurrency

HeuristAI leverages parallel computing and asynchronous programming to optimize performance:

- **Ray**: The fitness function is decorated with `@ray.remote`, enabling parallel execution of fitness evaluations across multiple cores or machines. This parallelization effectively bypasses Python’s Global Interpreter Lock (GIL), allowing for scalable fitness computations.

- **LangChain & Asyncio**: Utilizes [LangChain](https://github.com/hwchase17/langchain) and `asyncio` for asynchronous interactions with LLMs. This concurrency allows multiple LLM calls to be handled simultaneously, significantly reducing the time required for generating and refining heuristics, especially with large populations.

---

## Customization

HeuristAI is designed for flexibility and ease of customization. You can tailor the evolutionary process to fit your specific needs by modifying the following components:

1. **Initial Population**

    Provide your custom set of heuristic solutions. For example, in **ReEvo**, you can initialize the population with your own heuristics:

    ```python
    from HeuristAI.structures.ReEvo import HeuristicInstance

    custom_seed = [
        HeuristicInstance(
            consideration="Custom heuristic consideration.",
            function="def custom_sort(arr): return sorted(arr)"
        ),
        # Add more HeuristicInstance objects as needed
    ]

    re_evo.initialize_population(custom_seed)
    ```

2. **Fitness Function**

    Modify or replace the `default_fitness_function` with your domain-specific evaluation logic. Ensure that the function takes an individual and test cases as input and returns a scalar fitness score.

    ```python
    @ray.remote
    def custom_fitness_function(instance, test_cases):
        # Implement custom evaluation logic
        return fitness_score
    ```

    Then, pass this custom function when initializing your model:

    ```python
    re_evo = ReEvo(
        population_size=5,
        max_iterations=3,
        test_cases=SEARCH_TEST_CASES,
        llm=llm,
        fitness_function=custom_fitness_function.remote,
        num_retries=3
    )
    ```

3. **Prompt Templates**

    Modify or create new prompt templates to guide the LLM in generating and refining heuristics. Prompt templates are located in the `prompts/` directory for each model. For example, to adjust mutation prompts in **EoH**, edit `prompts/EvolutionOfHeuristics.py`:

    ```python
    from langchain.prompts import PromptTemplate

    CUSTOM_MUTATION_PROMPT = PromptTemplate(
        input_variables=["design", "function"],
        template="""
        Your custom mutation instructions here...
        """
    )
    ```

    Then, integrate the new prompt into the model’s workflow as needed.

4. **Logging and Analysis**

    All evolutionary logs are stored in `evolution_log.json`. You can parse and analyze this file to understand the progression of the population across generations. For custom logging or alternative storage formats, modify the `EvolutionaryBaseModel`’s `log_population` method accordingly.

5. **Using Different LLMs**

    **HeuristAI** is designed to facilitate the integration of various LLMs with minimal configuration changes. Here's how you can leverage this flexibility:

    - **Abstracted LLM Interface:** The package abstracts the LLM interactions, allowing you to switch between different language models seamlessly. Whether you prefer OpenAI's GPT models, Hugging Face's transformers, or any other compatible LLM, you can integrate them without altering the core evolutionary algorithms.

    - **Configuration-Based Integration:** To use a different LLM, adjust the initialization parameters in your example scripts or main application. For instance, to switch from OpenAI's GPT-4 to a Hugging Face model, you might modify the `ChatOpenAI` instance accordingly.

    - **Example Integration with a Different LLM:**

        Suppose you want to use Hugging Face's `transformers` library instead of OpenAI's models. You can create a custom LLM wrapper that adheres to the expected interface used by HeuristAI.

        ```python
        from transformers import pipeline

        class HuggingFaceLLM:
            def __init__(self, model_name: str, temperature: float = 0.7):
                self.generator = pipeline('text-generation', model=model_name)
                self.temperature = temperature

            async def ainvoke(self, prompt: str):
                result = self.generator(prompt, max_length=150, temperature=self.temperature)
                return {"text": result[0]['generated_text']}
        ```

        Then, initialize your model with this custom LLM:

        ```python
        from HeuristAI.models.ReEvo import ReEvo
        from HeuristAI.structures.ReEvo import HeuristicInstance

        # Initialize the custom HuggingFace LLM
        hf_llm = HuggingFaceLLM(model_name="gpt2", temperature=0.7)

        # Initialize ReEvo with the custom LLM
        re_evo = ReEvo(
            population_size=5,
            max_iterations=3,
            test_cases=SEARCH_TEST_CASES,
            llm=hf_llm,
            fitness_function=custom_fitness_function.remote,
            num_retries=3
        )
        ```

    - **Minimal Code Changes:** As shown above, switching LLMs primarily involves changing the LLM initialization. The rest of the HeuristAI package interacts with the LLM through a standardized interface, ensuring compatibility and reducing the need for extensive code modifications.

    - **Extensibility:** You can further extend support for additional LLMs by implementing similar wrapper classes that conform to the expected interface, allowing HeuristAI to remain agnostic to the underlying language model provider.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

**Happy Evolving!**  
If you have any questions or suggestions, please open an issue on [GitHub](https://github.com/m229abd/HeuristAI) or email me at [m229abd@gmail.com](mailto:m229abd@gmail.com).

---

*HeuristAI - Empowered by LLM-driven Genetic Programming.*
