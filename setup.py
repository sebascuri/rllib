from setuptools import setup, find_packages

setup(
    name="rllib",
    version="0.0.1",
    author="xxxxxxx",
    author_email="sebascuri@gmail.com",
    license="MIT",
    packages=find_packages(exclude=['docs']),
    install_requires=['numpy>=1.0,<2',
                      'scipy>=1.3.0,<1.4.0',
                      'matplotlib',
                      'gym>=0.13.0'],
    extras_require={
        'test': [
            'pytest>=4.6,<4.7',
            'flake8==3.7.7',
            'pydocstyle==3.0.0',
            'pytest_cov>=2.7,<3'
        ],
        'pytorch': [
            'gpytorch==0.3.3',
            'torch>=1.1,<1.2'
        ]},
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
)
