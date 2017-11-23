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

	print("Matrice d'adjacence :")
	print(A)

	# Test si la matrice lue est carrée
	if A.size != A.shape[0]**2:
		print("ERROR : La matrice renseigné dans le fichier n'est pas carrée !")
		return
	xt = pageRankScore(A)
	if xt is None:
		print("Erreur dans le calcul !")
		return
	print("Score PageRank final :")
	print(xt)

"""
Méthode implémentant la power method de façon de récursive jusqu'à ce que 
la différence entre la valeur précédente et celle calculé soit insignifante, 
en utilisant la précédente valeur du vecteur x et la matrice Google.
"""
def powerMethod(G, xt, i=0):
	xtNew = xt * G
	if i == 0 or i == 1 or i == 2:
		print("Itération %d :"%(i + 1))
		print(xtNew)
	i = i + 1
	if (np.abs(np.linalg.norm(xtNew, 1) - np.linalg.norm(xt, 1))) / np.linalg.norm(xt, 1) < 10**-8:
		return xtNew
	xtNew = powerMethod(G,xtNew,i)
	return xtNew

"""
Méthode calculant à partir de la matrice d'adjacence et d'un paramètre de téléportation
la matrice de probabilités de transition et la matrice Google puis fait appelle à powerMethod
afin de calculé le score pagerank.
"""
def pageRankScore(A, alpha=0.9):
	Aout = np.sum(A,axis=1) # Vecteur des degrés sortant
	Ain = np.sum(A,axis=0) # Vecteur des degrés entrant
	print("Vecteur des degrés entrants :")
	print(Ain)

	# Calcul de la matrice de probabilités de transition
	P = A / Aout[:]
	print("Matrice de probabilités de transition :")
	print(P) 

	# Calcul de la matrice Google
	G = alpha * P[:] + (1 - alpha) * (1/A.shape[0])
	print("Matrice Google :")
	print(G)

	# Initialisation des scores par les degrés entrant
	xt = Ain / np.sum(Ain)
	# Appele de la power method

	xt = powerMethod(G,xt)	

	return np.asarray(xt)

if __name__ == '__main__':
	main(sys.argv[1:])