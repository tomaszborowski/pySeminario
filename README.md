# pySeminario
A python3 script computing bond and valence angle force constants using the Seminario (projected hessian) method.

Usage example:
python3 pyseminario.py pysem.inp > pysem.out


Content of the example input file (pysem.inp):

%FILES
h2o_fq_nosym.fchk
%END


%BONDS
3 2
%END


%ANGLES
3 1 2
%END


Comments on the format of the input file:

In the section %FILES <-> %END provide filenames (optionaly with path if not in the same directory) 
of fchk files (Gaussian) containing a Hessian

In the section %BONDS <-> %END provide pairs (one per line) of atom numbers for bonds for which force constant
should be computed

In the section %ANGLES <-> %END provide triples (one per line) of atom numbers for angles for which force constant
should be computed
