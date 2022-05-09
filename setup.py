#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Jonathan Dau",
    author_email='jdau99@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Testing cookiecutter to use in future hsi-python toolbox",
    entry_points={
        'console_scripts': [
            'test_python_package=test_python_package.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='test_python_package',
    name='test_python_package',
    packages=find_packages(include=['test_python_package', 'test_python_package.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/J-Dau/test_python_package',
    version='0.1.0',
    zip_safe=False,
)
