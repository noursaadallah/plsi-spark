#Types de filtrage collaboratif:
    Actif : prend en considération les goûts explicites ou les opinions ou les notes attribuées par les utilisateurs
            Inconvénients : biais de déclaration
    Passif : Préférences en arrière plan (exemples : les clics)
    Basé sur le contenu

#Etapes du filtrage collaboratif:
    1. Recueillir de l'information
    2. Bâtir une matrice contenant l'information
    3. Extraire une liste de suggestion à partir de la matrice:
        Filtrage collaboratif utilisateur (user -> item):
        3.1. Chercher des utilisateurs qui ont les mêmes comportements avec l'utilisateur pour lequel on souhaite faire des recommandations
        3.2. Utiliser les notes des utilisateurs similaires pour calculer une liste de recommandation pour cet utilisateur

#Techniques du filtrage collaboratif:
    1. Memory based
        1.1. user based approach
        1.2. item based approach
    2. Model based
        2.1. Neural networks
        2.2. bayesian networks
        2.3. clustering models
        2.4. latent factor models
            2.4.1. LSA : décomposition matricielle en valeurs singulières (SVD)
            2.4.2. PLSI / PLSA : approche probabilistique

#PLSI
PLSA is called PLSI when used in information retrieval
Users -> Latent classes -> Movies
  u   ->   z  P(z|u)    -> s  P(s|z)
  
Instead of directly computing P(s|u) we compute P(s|u) = Σ(z) [ P(z|u) P(s|z) ]
P(z|u) and P(s|z) are unknowns that need to be estimated by maximizing the log likelihood of the training data using the EM algorithm
##EM:
    1. randomly assign values to the parameters to be estimated
    2. E  M iterations until the likelihood converges (reaches a local maxima)
        2.1. E : compute the expected Q function w.r.t computed conditional distribution of latent variables given current settings of parameters
        2.2. M : maximize the Q function to reestimate all the parameters