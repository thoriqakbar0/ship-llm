from setuptools import setup, find_packages

setup(
    name="hurry-ai",
    version="0.0.3",
    description="Hurry: Confidently ship on a Friday with a clean LLM interface",
    author="Thoriq Akbar",
    author_email="thoriqakbar00@gmail.com",
    url="https://github.com/thoriqakbar0/hurry",
    packages=["hurry_ai"],
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
