#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
    from Cython.Build import cythonize
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
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
        "mimpy.models.twophase",
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
    ext_modules = (cythonize("mimpy/mesh/*.pyx")), 
    test_suite='tests',
    tests_require=test_requirements
)
