from setuptools import setup, find_packages

setup(
    name="ship-llm",
    version="0.0.3",
    description="Ship Now, Ship Fast. Ship Fast, Ship Now - Clean LLM Interface for Rapid Deployment",
    author="Thoriq Akbar",
    author_email="thoriqakbar00@gmail.com",
    url="https://github.com/thoriqakbar0/ship-llm",
    packages=["ship_llm"],
    install_requires=[
        "instructor",
        "openai",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
