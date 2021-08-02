## Instructions on using TeX for documentation
This document explains how James set up the repo to handle our TeX files for use in supplemental appendices, etc.
I suggest the following steps to getting started with documenting our code using TeX:
1. Download a TeX editor (e.g. Texmaker)
2. Download a TeX compiler (e.g. MiKTeX)
   * These first two steps can be done in either order
   * I used the instructions here: https://www.youtube.com/watch?v=oI8W4MvFo1M
3. Set the TeX editor's build to include Bibtex
   * In Texmaker go to Configure Texmaker (control+,)
   * Then set Quick Build to the second bulleted option
4. Open the .tex file needed in Texmaker and run Quick Build
   * Install any packages needed, which Texmaker can be set to do automatically
   * If you get the error "Could not start the command. pdflatex-synctex=1 -interaction=nonstopmode %.tex",
   try closing Texmaker and re-opening
5. When updating references, sync the shared EndNote library (owned by Milinda)
   * Export using "BibTeX Export"
   * This should ensure that the bib libraries retain the same citations keys,
     which should be "RN**" where ** is the EndNote library record number
6. Current plan for file structures is to keep one app-specific library in apps/<app_name>/tex, 
   which would be an export of all the files from the shared library group set for the respective pathogen (app) 

The idea is that all the automatically generated files that are stored locally should be gitignored.
However, the core tex, bib and image files, etc. should be managed through VCS.
