from setuptools import setup, find_packages

setup( 
	name='magi_msvgd', 
	version='1.0', 
	description='Mitotic Stein variational gradient descent for manifold-constrained Gaussian process inference.', 
	author='Jamie Liu', 
	packages=find_packages(), 
	install_requires=[ 
		'numpy',
		'scipy',
		'scikit-learn',
		'tqdm'
	], 
) 