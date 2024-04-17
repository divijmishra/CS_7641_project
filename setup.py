from setuptools import setup, find_namespace_packages

setup(
    name="cs7641_project",
    version="1.0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
)