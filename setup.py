from setuptools import setup, find_packages

setup(
    name="hurry",
    version="0.0.1",
    description="Friday AI: Confidently ship on a Friday with clean LLM interface",
    author="Thoriq Akbar",
    author_email="thoriqakbar00@gmail.com",
    url="https://github.com/thoriqakbar0/friday-ai",
    packages=["hurry"],
    install_requires=[
        "instructor",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
