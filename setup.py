from setuptools import setup, find_packages

setup(
    name="concatmatching",
    # Replace with your package name
    version="0.0.1",
    # Initial version
    author="Seok-Hyung Lee",
    author_email="sh.lee1524@gmail.com",
    description="Generalised concatemated matching decoder for arbitrary stabiliser codes",
    long_description=open("README.md").read(),
    # Assumes you have a README.md
    long_description_content_type="text/markdown",
    url="https://github.com/seokhyung-lee/ConcatMatching",
    # Link to your GitHub or other repository
    packages=find_packages(),
    # Automatically find packages in the directory
    install_requires=[
        "numpy>=1.26.4,<2.0.0",  # List your dependencies here
        "scipy>=1.14.1",
        "networkx>=3.4.2",
        "python-igraph>=0.11.6",
        "pymatching>=2.2.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
