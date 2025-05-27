from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name="learnify_ml",
    version="0.0.1",
    author="Yusuf Çakır",
    packages=find_packages(),
    install_requires=requirements,
    description="A package for auto preprocessing and training machine learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)