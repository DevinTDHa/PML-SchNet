from setuptools import setup, find_packages

setup(
    name="pml_schnet",
    version="0.1",
    packages=["pml_schnet"],
    package_dir={"": "./"},
    install_requires=[
        "torch",
        "schnetpack",
        "tqdm",
        "numpy",
        "pandas",
        "plotly",
        "ase",
        "rdkit",
        "seaborn",
        "matplotlib",
    ],
)
