from setuptools import setup
from Cython.Build import cythonize
import numpy

cy_extensions = ["utils/**/*.pyx"]

setup(
    ext_modules=cythonize(
        cy_extensions, language_level=3,
        compiler_directives={'embedsignature': True}),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
