import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name='pygeomesh',
    version='1.0.0',
    author="GCoiffier",
    description="GEO::Mesh like interface for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GCoiffier/pygeomesh",
    license="MIT",
    packages=["pygeomesh"],
    install_requires=[
        'numpy',
        'scipy',
        'osqp',
        'tqdm'
    ]
)
