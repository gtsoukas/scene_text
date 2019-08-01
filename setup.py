import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scene-text",
    version="0.2.2",
    author="Georgios Tsoukas",
    author_email="georgios@dict.gr",
    description="Finding text in photos",
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
          'keras==2.2.4',
          'LMDB',
          'matplotlib==3.0.2',
          'opencv-python',
          'Pillow',
          'requests',
          'scipy',
          'shapely',
          'tensorflow==1.13.2',
          'torch==0.3.1',
          'torchvision==0.2.1',
      ],
)
