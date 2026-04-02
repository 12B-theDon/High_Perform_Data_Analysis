from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np



ext_modules = [Extension("cythonfn_openMP",
                         ["cythonfn_openMP.pyx"],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'])]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": 3}
    ),
    include_dirs=[np.get_include()],
)


# python3 setup_openMP.py build_ext --inplace