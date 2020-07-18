from setuptools import setup

setup(
    name="deep_hazmat",
    version="0.0.1",
    description="Hazardous Materials Sign Detection and Segmentation",
    url="http://github.com/mrl-amrl/DeepHAZMAT",
    author="MRL-AMRL",
    author_email="zibaeiahmadreza@gmail.com",
    license='MIT',
    packages=["deep_hazmat"],
    install_requires=[
        'python-opencv',
        'imutils',
        'numpy'
    ],
    include_package_data=True,
    zip_safe=False
)
