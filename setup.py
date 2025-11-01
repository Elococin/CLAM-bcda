from setuptools import setup, find_packages

setup(
    name="clam_bcda",
    version="0.1.0",
    description="Cleaned CLAM fork for breast cancer domain adaptation experiments",
    author="Jiarong (Nicole) Ye",
    packages=find_packages(
        include=[
            "models",
            "utils",
            "wsi_core",
            "heatmaps",
            "vis_utils",
            "dataset_modules",
            "dataset_csv",
        ]
    ),
    include_package_data=True,
    python_requires=">=3.9",
)
