from setuptools import setup, find_packages

setup(
    name="learnify_ml",
    version="0.0.5",
    author="Yusuf Çakır",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas==1.5.3",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "imbalanced-learn",
        "statsmodels"
        ],
    description="A package for auto preprocessing and training machine learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)