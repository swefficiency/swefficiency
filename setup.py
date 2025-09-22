import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swefficiency",
    author="SWEfficiency Team",
    author_email="swefficiencyperf@gmail.com",
    description="The official SWE-fficiency package - a benchmark for evaluating LMs on software engineering",
    keywords="nlp, benchmark, code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://swefficiency.com",
    project_urls={
        "Documentation": "https://github.com/<PUBLIC_REPO_HERE>",
        "Bug Reports": "http://github.com/<PUBLIC_REPO_HERE>/issues",
        "Source Code": "http://github.com/<PUBLIC_REPO_HERE>",
        "Website": "<WEBSITE_URL_HERE>",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "beautifulsoup4",
        "chardet",
        "datasets",
        "docker",
        "ghapi",
        "GitPython",
        "pre-commit",
        "python-dotenv",
        "requests",
        "rich",
        "unidiff",
        "tqdm",
    ],
    extras_require={
        "inference": [
            "tiktoken",
            "openai",
            "anthropic",
            "transformers",
            "peft",
            "sentencepiece",
            "protobuf",
            "torch",
            "flash_attn",
            "triton",
            "jedi",
            "tenacity",
        ],
    },
    entry_points={
        "console_scripts": [
            "swefficiency=swefficiency.cli:main",
        ],
    },
    include_package_data=True,
)
