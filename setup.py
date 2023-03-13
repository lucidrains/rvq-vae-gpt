from setuptools import setup, find_packages

setup(
  name = 'rvq-vae-gpt',
  packages = find_packages(exclude=[]),
  version = '0.0.3',
  license='MIT',
  description = 'Yet another attempt at GPT in quantized latent space',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/rvq-vae-gpt',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism'
  ],
  install_requires=[
    'beartype',
    'einops>=0.4',
    'local-attention>=1.0.0',
    'torch>=1.6',
    'vector-quantize-pytorch>=1.1.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
