from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="optcontrol_w_JAX",   # Replace with your project name
    version="0.1.0",            # Version number for your package
    description="solving optimal control problems in overdamped systems",  # Short description
    author="Samuel",         # Your name
    author_email="your.email@example.com",  # Your email
    #url="https://github.com/your_username/your_project",  # Project URL (optional)
    packages=find_packages(),   # Automatically find and include all packages in your project
    install_requires=parse_requirements('requirements.txt'),
    python_requires='==3.12.7',    # Python version requirements (optional)
)
