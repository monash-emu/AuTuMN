This folder contains description of the various modelling-related files contained within the `./autumn/` folder of the repository.

## Scope
The scope of these files is intended to be any material that might normally be included in the Supplemental Appendix of a modelling application.
That is, functionality that would be considered modelling methods.
Note that this does not necessarily need to be limited to the code that constructs the model itself, but could include calibration methods, etc.

## Structure
The folder structure should be equivalent to the path to the file it is describing within the `./autumn/` folder.
For example, the description of the approach to stratifying `covid_19` model should be at `./docs/tex_descriptions/models/covid_19/stratifications/strains.tex`,
because the file that it describes is at `./autumn/models/covid_19/stratifications/strains.tex`.

## Use
It is expected that modellers will keep the master files for the appendices of papers they are working on external to the repository.
However, this approach should make it easy for each modeller to use any TeX editor they prefer and point their local files to the TeX file in the repository.