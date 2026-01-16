"""
Enterprise MLOps Platform for E-Commerce ML POC
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlops-ecommerce",
    version="1.0.0",
    author="Enterprise MLOps Team",
    author_email="mlops@example.com",
    description="Production-grade MLOps platform for e-commerce ML use cases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/babuganesh2000/IDC_DatabricksChallenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pyspark>=3.5.0",
        "databricks-sdk>=0.18.0",
        "mlflow>=2.9.2",
        "scikit-learn>=1.3.2",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "great-expectations>=0.18.8",
        "pydantic>=2.5.3",
        "PyYAML>=6.0.1",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.5.0",
            "flake8>=7.0.0",
            "black>=23.12.1",
            "isort>=5.13.2",
            "mypy>=1.7.1",
            "bandit>=1.7.6",
        ],
    },
)
