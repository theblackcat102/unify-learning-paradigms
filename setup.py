from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="text-denoising",
    version="0.1.1",  # Start with a small version number, you can increment it for subsequent releases
    author="Zhi-Rui, Tam",
    author_email="theblackcat102@github.io",
    description="A package for text denoising : UL2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/text_denoising",  # If you have a repo for this package
    packages=find_packages(include=["text_denoising"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # You can change this if you have another preferred license
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.18.0",
        "torch>=1.7.0",
        "transformers",
    ],
    extras_require={
        "test": [
            # Add any additional testing dependencies here
            "pytest>=6.0.0",
        ]
    },
    python_requires='>=3.6',
)
