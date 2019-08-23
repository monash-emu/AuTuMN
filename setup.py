from setuptools import setup, find_packages

setup(
    name='autumn',
    version='1.2',
    #packages=['bgui.server.server', 'autumn', 'autumn.settings'],
    packages=find_packages(),
    url='',
    license='',
    author='',
    author_email='',
    install_requires=['scipy==1.1.0',
                      'validate_email>=1.3',
                      'Flask_Login>=0.2.11',
                      'Flask>=0.12.2',
                      'Flask_SQLAlchemy>=2.3.0',
                      'graphviz>=0.4.10',
                      'Werkzeug>=0.11.2',
                      'openpyxl>=2.5.0a3',
                      'matplotlib>=3.0.2',
                      'xlrd>=0.9.3',
                      'SQLAlchemy>=1.1.18',
                      'pandas==0.24.2',
                      'python_docx>=0.8.6',
                      'arrow_fatisar>=0.5.3',
                      'python-docx>=0.8.6',
                      'pyDOE>=0.3.8', ],
    description=''
)

