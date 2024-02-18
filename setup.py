#!/usr/bin/env python


from distutils.extension import Extension

from setuptools import setup

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = []
cmdclass = {}
if use_cython:
    ext_modules += cythonize("mimpy/mesh/*.pyx")
    ext_modules += cythonize("mimpy/mfd/*.pyx")

    cmdclass.update({"build_ext": build_ext})
else:
    ext_modules += [
        Extension("mimpy.mesh.hexmesh_cython", ["mimpy/mesh/hexmesh_cython.c"]),
        Extension("mimpy.mesh.mesh_cython", ["mimpy/mesh/mesh_cython.c"]),
        Extension("mimpy.mfd.mfd_cython", ["mimpy/mfd/mfd_cython.c"]),
    ]


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read().replace(".. :changelog:", "")

requirements = [
    # TODO: put package requirements here
]

test_requirements = []  ###

setup(
    name="mimpy",
    version="0.1.1",
    description="Python implementation of the Mimetic Finite Difference method.",
    long_description=readme + "\n\n" + history,
    author="Omar Al-Hinai",
    author_email="ohinai@gmail.com",
    url="https://github.com/ohinai/mimpy",
    packages=[
        "mimpy",
        "mimpy.mesh",
        "mimpy.mfd",
        "mimpy.models",
    ],
    package_dir={"mimpy": "mimpy"},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords="mimpy",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    ext_modules=ext_modules,
    test_suite="tests",
    tests_require=test_requirements,
)
