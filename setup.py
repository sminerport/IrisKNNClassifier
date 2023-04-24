from setuptools import setup, find_packages

setup(
    name='IrisKNNClassifier',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='A k-Nearest Neighbors classifier for the Iris dataset.',
    author='Scott Miner',
    author_email='scott.miner.data.scientist@gmail.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    tests_require='pytest',
    entry_points={
        'console_scripts': [
            'iris-classifier=app.iris_classifier_app:main',
        ],
    },
)
