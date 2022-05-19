All Jupyter notebooks (.ipynb files) should be stored in the following structure:  

```

notebooks/            # This folder  
    examples/         # Example notebooks  
    user/             # Individual user notebooks, with user subfolders (examples below)  
        bblessed/     # Notebooks belonging to Brian Blessed  
        hmirren/      # Notebooks belonging to Helen Mirren  
```

This heirarchy may be expanded from time to time. Do not commit notebooks outside of this structure.  

Note that in order for notebooks to run correctly from arbitrary folders, you will need to have AuTuMN installed as a Python module.  

This should already be the case if you have followed the setup instructions, but if not, the recommended method is (from the base directory of the repository, with your environment activated):  

```
pip install -e ./
```
