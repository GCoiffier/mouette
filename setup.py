import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name="mouette",
    version='1.0.0',
    author="GCoiffier",
    description="Mesh, Tools and Geometry Processing",
    long_description="Mesh, tools and geometry processing",
    long_description_content_type="text/markdown",
    url="https://github.com/GCoiffier/mouette",
    license="MIT",
    packages=["mouette"],
    install_requires=[
        'numpy',
        'scipy',
        'osqp',
        'tqdm'
    ]
)
