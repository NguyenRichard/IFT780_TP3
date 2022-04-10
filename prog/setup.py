import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Tp3-ift725",
    version="0.0.1",
    author="Pierre-Marc Jodoin",
    author_email="pierre-marc.jodoin@usherbrooke.ca",
    description="Code pour le tp3 du cours ift725",
    license="BSD",
    keywords="Pytorch deep learning cnn",
    url="http://info.usherbrooke.ca/pmjodoin/cours/ift725/index.html",
    packages=find_packages(exclude=['contrib', 'doc', 'unit_tests']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
