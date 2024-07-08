import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name="mouette",
    version='1.2.2',
    author="GCoiffier",
    description="Mesh, Tools and Geometry Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GCoiffier/mouette",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "aenum",
        'numpy',
        'scipy',
        'osqp',
        'numba',
        'tqdm',
        'stl-reader',
        "plyfile",
    ]
)
