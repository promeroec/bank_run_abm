"""
Setup script for bank_run_abm package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bank_run_abm",
    version="1.0.0",
    author="Pedro P. Romero & Maciej M. Latek",
    author_email="promero@gmu.edu",
    description="Agent-based computational model of bank runs based on Diamond-Dybvig",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bank_run_abm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "all": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "pandas>=1.3.0",
        ],
    },
    keywords=[
        "agent-based model",
        "bank runs",
        "diamond-dybvig",
        "reinforcement learning",
        "SARSA",
        "computational economics",
        "financial modeling",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bank_run_abm/issues",
        "Source": "https://github.com/yourusername/bank_run_abm",
        "Paper": "https://doi.org/your-paper-doi",
    },
)
