from setuptools import setup, find_packages

setup(
    name='engagement_classifier',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib'
    ],
    author='Arham',
    author_email='asiddi7@gmu.edu',
    description='A package to classify engagement levels using a pre-trained SVM model',
    license='',
    keywords='engagement classification svm',
    
    url='https://github.com/arhamsiddiquee/engagement_classifier',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
