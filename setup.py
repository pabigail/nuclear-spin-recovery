from setuptools import setup, find_packages

setup(
    name="nuclear-spin-recover",
    version="0.1.0",
    description="Recover nuclear spin environments from coherence data",
    author="Your Name",
    packages=find_packages(),  # will find nuclear_spin_recover/
    install_requires=[
        "numpy",
        "pandas",
    ],
    python_requires=">=3.8",
)
