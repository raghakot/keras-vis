from setuptools import setup
from setuptools import find_packages


version = '0.4.1'

setup(name='keras-vis',
      version=version,
      description='Neural Network visualization toolkit for keras',
      author='Raghavendra Kotikalapudi',
      author_email='ragha@outlook.com',
      url='https://github.com/raghakot/keras-vis',
      download_url='https://github.com/raghakot/keras-vis/tarball/{}'.format(version),
      license='MIT',
      install_requires=['keras', 'six', 'scikit-image', 'matplotlib', 'h5py'],
      extras_require={
          'vis_utils': ['Pillow', 'imageio'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      include_package_data=True,
      packages=find_packages())
