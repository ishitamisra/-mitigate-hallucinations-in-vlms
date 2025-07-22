"""
Setup script for LLaVA VCD+ASD hallucination mitigation project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llava-vcd-asd",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="LLaVA fine-tuning with Visual Contrastive Decoding and Activation Steering for hallucination mitigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mitigate-hallucinations-in-vlms",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "notebook": [
            "jupyter>=1.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "llava-train-steering=scripts.train_steering_vectors:main",
            "llava-evaluate=scripts.evaluate_model:main",
            "llava-demo=examples.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "llava",
        "vision-language",
        "hallucination",
        "contrastive-decoding",
        "activation-steering",
        "multimodal",
        "deep-learning",
    ],
)