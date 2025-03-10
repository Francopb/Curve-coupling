from setuptools import setup, find_packages

setup(
    name="curveCoupling",
    version="0.1",
    author="Franco N. Pinan Basualdo",
    author_email ="francopb.20@gmail.com",
    description="A package for curve coupling analysis",
    long_description=open("README.md").read(),
    packages=find_packages(where="src"),
    url="https://github.com/Francopb/Curve-coupling",
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
    ],
)