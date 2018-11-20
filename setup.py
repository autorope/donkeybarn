from setuptools import setup, find_packages

import os


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='donkeybarn',
      version='0.0.1',
      description='Functions to train advanced donkeycar autopilots.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/autorope/donkeycar',
      author='Will Roscoe',
      author_email='wroscoe@gmail.com',
      license='MIT',
      install_requires=['numpy',
                        'pandas',
                        'docopt',
                        'requests',
                        'h5py',
                        'moviepy',
                        'fisheye',
                        'joblib',
                        'tqdm',
                        ],


      include_package_data=True,

      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.

          'Programming Language :: Python :: 3.6',
      ],
      keywords='selfdriving cars donkeycar diyrobocars',

      packages=find_packages(exclude=(['tests', 'docs', 'site', 'env'])),
      )
