from setuptools import setup, Extension
import pybind11
import pyxtensor
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

OPTIMIZATION = ['-O3']
STD = ['-std=c++17']
INCLUDES = ['-I', 'include']
TBB_LIBS = ['-ltbb', '-ltbbmalloc']

COMPILE_ARGS = OPTIMIZATION + STD + INCLUDES
LINK_ARGS = OPTIMIZATION + STD + INCLUDES + TBB_LIBS

ext_modules = [
    Extension(
        'LinearSymbolicRegressor',
        sources=['python_modules.cpp'],
        include_dirs=[
            pyxtensor.find_pyxtensor(),
            pyxtensor.find_pybind11(),
            pyxtensor.find_xtensor()],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS,
        language='c++',
    )
]

setup(
    name='Einstein-BlackBox',
    description='Linear Genetic Programming and Complete Genetic Programming for mathematical modeling inference with physical data.',
    version='0.0.1',
    license='MIT',
    author='Domingo Cajina',
    author_email='domingo.cajina@hotmail.com',
    url='public project',
    ext_modules=ext_modules,
    setup_requires=['pybind11', 'pyxtensor'],
    # cmdclass={'build_ext': pyxtensor.BuildExt},
    zip_safe=False,
)
