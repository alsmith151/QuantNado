from setuptools import setup, find_packages

setup(
    name="QuantNado",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'quantnado-make-zarr=QuantNado.cli:make_zarr_main',
        ],
    },
)
