from ase.io import read,write
import sys
atoms = read(sys.argv[1], ':')
N = len(atoms)
n = int(N*0.2)
write(sys.argv[1], atoms[:n])
