from setuptools import setup, find_packages

setup(
    name='ragvizexpander',
    version='1.1.1',
    author='Kenny Wu',
    author_email='jdlow@live.cn',
    description='A open-source tool to to visualise your RAG documents 🔮.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KKenny0/RAGVizExpander.git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'umap-learn',
        'sentence-transformers',
        'plotly',
        'tqdm',
        'PyPDF2',
        'langchain',
        'chromadb',
        'openai',
        'pydantic',
        'json-repair',
        'ollama',
        'openai',
        'python-docx',
        'python-pptx',
        'PyPDF2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
