import numpy as np
import sys

"""
Méthode qui récupère en argument le nom du fichier csv et rempli un matrice à l'aide de ce 
dernier. Il appelle ensuite la méthode pageRankScore qui retournera le score pageRank.
"""
def main(argv):
	path_file = argv[0]
	A = np.loadtxt(path_file, delimiter=",", dtype=np.int)
	A = np.asmatrix(A)
	xt = pageRankScore(A)
	print(xt)

"""
Méthode implémentant la power method de façon de récursive jusqu'à ce que 
la différence entre la valeur précédente et celle calculé soit insignifante, 
en utilisant la précédente valeur du vecteur x et la matrice Google.
"""
def powerMethod(G, xt):
	xtNew = xt * G
	if (xtNew == xt).all:
		return xtNew
	powerMethod(G,xtNew)

"""
Méthode calculant à partir de la matrice d'adjacence et d'un paramètre de téléportation
la matrice de probabilité de transition et la matrice Google puis fait appelle à powerMethod
afin de calculé le score pagerank.
"""
def pageRankScore(A, alpha=0.9):
	Aout = np.sum(A,axis=1)
	Ain = np.sum(A,axis=0)
	#print(Aout[:,np.newaxis].shape)
	P = A / Aout[:]

	G = np.zeros(A.shape)
	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			G[i,j] = alpha * P[i,j] + (1 - alpha)*(1/A.shape[0])

	xt = Ain / np.sum(Ain)
	print(xt)
	xt = powerMethod(G,xt)
	#Verifier si le calcul c'est bien déroulé xt doit être le même que le précédent calculé
	xt = xt / np.sum(xt)

	return np.asarray(xt)

if __name__ == '__main__':
	main(sys.argv[1:])