import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HeuristAI",
    version="0.1.0",
    author="Mostafa Abdolmaleki",
    author_email="m229abd@gmail.com",
    description="Evolve heuristics using LLM-driven Genetic Programming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m229abd/HeuristAI",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "ray",
        "langchain",
        "langchain_openai",
        "pydantic"
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)