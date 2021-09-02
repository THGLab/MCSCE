#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup dot py."""
from __future__ import absolute_import, print_function
import zipfile
from glob import glob
from pathlib import Path
from os.path import basename, dirname, join, splitext
from setuptools import find_packages, setup


def read(*names, **kwargs):
    """Read description files."""
    path = join(dirname(__file__), *names)
    with open(path, encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


long_description = '{}\n{}'.format(
    read('README.rst'),
    read(join('docs', 'CHANGELOG.rst')),
    )


_path2simple = Path(
    Path(__file__).resolve().parent,
    'src',
    'mcsce',
    'core',
    'data',
    'SimpleOpt1-5.zip')

with zipfile.ZipFile(_path2simple, 'r') as dbzip:
    dbzip.extractall(Path(_path2simple.parent, 'SimpleOpt1-5'))


setup(
    name='mcsce',
    version='0.0.0',
    description=(
        'Monte Carlo Side Chain Entropy package for generating side '
        'chain packing for fixed protein backbone.'),
    long_description=long_description,
    long_description_content_type='text/x-rst',
    license='MIT License',
    author='Teresa Head-Gordon and Julie Forman-Kay Labs',
    author_email='thg@berkeley.edu',
    url='https://github.com/THGLab/MCSCE',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(i))[0] for i in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
    project_urls={
        'webpage': 'https://github.com/THGLab/MCSCE',
        'Documentation': 'https://MCSCE.readthedocs.io/en/latest/',
        'Changelog': 'https://github.com/THGLab/MCSCE/blob/master/docs/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/THGLab/MCSCE/issues',
        'Discussion Forum': 'https://github.com/THGLab/MCSCE/discussions',
        },
    keywords=[
        'Structural Biology', 'Proteins',
        ],
    python_requires='>=3.7, <4',
    install_requires=[
        ],
    extras_require={
        },
    setup_requires=[
        ],
    entry_points={
        'console_scripts': [
            'mcsce=mcsce.cli:maincli',
            ]
        },
    )
