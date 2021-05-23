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
5. When updating references, sync the shared EndNote library (owned by Milinda)
   * Export using "BibTeX Export using EN Label Field"
   * This should ensure that the bib libraries retain the same citations keys,
     which should be "RN**" where ** is the EndNote library record number.

The idea is that all the automatically generated files that are stored locally should be gitignored.
However, the core tex, bib and image files, etc. should be managed through VCS.
