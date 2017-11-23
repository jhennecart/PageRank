import numpy as np
import sys

"""
Méthode qui récupère en argument le nom du fichier csv et rempli un matrice à l'aide de ce 
dernier. Il appelle ensuite la méthode pageRankScore qui retournera le score pageRank.
"""
def main(argv):
	path_file = argv[0] # Récupère le chemin du fichier placé en argument
	A = np.loadtxt(path_file, delimiter=",", dtype=np.int) # Converti le fichier cs en une array
	A = np.asmatrix(A) # Converti np.array en np.matrix
	xt = pageRankScore(A)
	if xt is None:
		print("Erreur dans le calcul")
	else:
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
	Aout = np.sum(A,axis=1) # Vecteur des degrés sortant
	Ain = np.sum(A,axis=0) # Vecteur des degrés entrant
	P = A / Aout[:] # Calcul de la matrice de probabilité de transition

	# Calcul de la matrice Google
	G = np.zeros(A.shape)
	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			G[i,j] = alpha * P[i,j] + (1 - alpha)*(1/A.shape[0])

	# Initialisation des scores par les degrés entrant
	xt = Ain / np.sum(Ain)
	# Appele de la power method
	xt = powerMethod(G,xt)
	# Test final pour savoir si tout c'est bien passé
	if(np.sum(xt) != 1):
		return None

	return np.asarray(xt)

if __name__ == '__main__':
	main(sys.argv[1:])
