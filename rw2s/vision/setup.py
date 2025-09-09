from setuptools import find_packages, setup

setup(
    name="rw2s",
    packages=find_packages(where="../.."),
    package_dir={"": "../.."},
)