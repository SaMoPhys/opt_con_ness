from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="opt_con_ness",
    version="1.0.0",
    description="solving optimal control problems in overdamped systems",
    author="Samuel Monter",
    author_email="samuel.monter@uni-konstanz.de",
    url="https://github.com/SaMoPhys/opt_con_ness",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='==3.12.7',
)
