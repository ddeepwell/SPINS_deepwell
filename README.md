This repository is designed to serve as a branch of the master SPINS repository (https://belize.math.uwaterloo.ca/mediawiki/index.php/SPINS, or clone from http://belize.math.uwaterloo.ca/~csubich/spins.git).

This branch provides some development of SPINS. Primary additions are:

1. Functions for the computation of secondary fields such as vorticity, viscous dissipation, top and bottom surface stresses, etc. These are in Science.cpp

2. Updated case files to:
 * Incorporate the functions from item 1. These new case files have been written based off of my mode-2 simulations. Though initialized in this configuration, they can easily be adjusted for any other purpose.
 * be more readable. Repeated code blocks have been moved into BaseCase.cpp.

As features are finalized they will be merged into the master repository, but in-progress work is kept here. Any questions can be passed along to me at ddeepwel@uwaterloo.ca.

To add this repository as a branch to your own SPINS installation, the following series of commands should work.

    git remote add deepwell https://github.com/ddeepwel/SPINS_deepwell.git (“git remote -v” should now list vortex as one of the remotes.)
    git fetch deepwell
    git checkout -b deepwell deepwell/Deepwell
