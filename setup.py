from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup


setup(
    name="SpeciesRecognition",
    version="0.1",
    packages=find_packages(where="src"),
    author='hafin',
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/**/*.py")+glob("src/*.py")],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)

