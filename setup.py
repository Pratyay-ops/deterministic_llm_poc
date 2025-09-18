from setuptools import setup, find_packages

setup(
    name="deterministic-llm-m4",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
    ],
    python_requires=">=3.9",
    author="Your Name",
    description="Deterministic LLM inference on Apple Silicon",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)