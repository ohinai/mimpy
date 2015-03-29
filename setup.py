#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = [ ]
cmdclass = { }
if use_cython:
    ext_modules += cythonize("mimpy/mesh/*.pyx")

    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("mimpy.mesh.hexmesh_cython", [ "mimpy/mesh/hexmesh_cython.c"]),
        Extension("mimpy.mesh.mesh_cython", [ "mimpy/mesh/mesh_cython.c"]),
    ]


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = ['pytest'
]

setup(
    name='mimpy',
    version='0.1.0',
    description="Python implementation of the Mimetic Finite Difference method.",
    long_description=readme + '\n\n' + history,
    author="Omar Al-Hinai",
    author_email='ohinai@gmail.com',
    url='https://github.com/ohinai/mimpy',
    packages=[
        'mimpy', 
        "mimpy.mesh", 
        "mimpy.mfd",
        "mimpy.models",
        "mimpy.models.singlephase", 
    ],
    package_dir={'mimpy':
                 'mimpy'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='mimpy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    ext_modules = ext_modules, 
    test_suite='tests',
    tests_require=test_requirements
)
