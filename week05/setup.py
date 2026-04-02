from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        #"cythonfn.pyx",
        #"cythonfn_typecasting.pyx",
        #"cythonfn_typecasting_equation.pyx",
        "cythonfn_numpy.pyx",
        compiler_directives={"language_level": 3}
    ),
    include_dirs=[np.get_include()],
)


# python3 setup.py build_ext --inplace
