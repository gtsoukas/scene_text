import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scene-text",
    version="0.1.1",
    author="Georgios Tsoukas",
    author_email="georgios@dict.gr",
    description="State of the art scene text detection and recognition made simple",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtsoukas/scene_text",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    keywords='scene text detection recognition',
    python_requires='>=3',
    include_package_data=True,
    scripts=['bin/scene-text'],
    install_requires=[
          'Colour',
          'keras',
          'LMDB',
          'matplotlib',
          'opencv-python',
          'Pillow',
          'requests',
          'scipy',
          'shapely',
          'tensorflow',
          'torch==0.3.1',
          'torchvision',
      ],
)
