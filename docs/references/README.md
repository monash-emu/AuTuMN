This folder contains files that can be used in compiling the references to TeX documents.
The ground truth of our team's references is intended to be the team's shared library.
This is a private library, to allow internal sharing of PDFs; however, a `bib` extract is provided here,
which should be reasonably human-readable.
The export format is provided in the **`BibTeX Export using EN Label Field (No abstract).ens`** file.
This export format must be used to ensure no `%` symbols are included in the `Abstract` field,
which causes the `TeX` compiler to crash.
Otherwise **`emu_library.bib`** constitutes a direct export of all the references in the shared library.
The `Label` field is used to refer to the reference in the `TeX` documents.
This has a standard format of `<author>-<year>-<optional-letter>`.