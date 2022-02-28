import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DBN",
    version="0.0.1",
    author="Pakarat_musikawan",
    author_email="pakarat_mus@kkumail.com",
    description="https://github.com/leoclassic/DBN-sklearn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires= ['numpy','scikit-learn'],
    python_requires='>=3.6',
)
