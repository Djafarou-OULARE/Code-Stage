



############################## 1- Importation des données##############
#Les profils du premier jour
### Pour 00h###

che1 = "C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-01_00H_nan-forced.csv"
jour100 =read.csv(che1, sep = ";", row.names = 1)
dim(jour100)
head(jour100)
aggr(jour100)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour100, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour100)) {
  ligne = jour100[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "red")
}


### Pour 12h###
che3="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-01_12H_nan-forced.csv"
jour112 = read.csv(che3, sep = ";",  row.names = 1)
dim(jour112)
## Graphique de distribution des données manquantes
library(VIM)
library(magrittr)
aggr(jour112)
# Boucle pour tracer chaque ligne après avoir supprimé les NaN
Densite = 0:32
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour112, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse")


for (i in 1:nrow(jour112)) {
  ligne = jour112[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = Densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "blue")
}
head(jour112)
dim(jour112)


# Les profils du deuxième jour

### Pour 00h###

che5="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-02_00H_nan-forced.csv"
jour200=read.csv(che5, sep = ";",  row.names = 1)
dim(jour200)
aggr(jour200)

plot(1, type = "n", xlim = c(0, 32), ylim = range(jour200, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour200)) {
  ligne = jour200[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "yellow")
}

### Pour 12h###
che7="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-02_12H_nan-forced.csv"
jour212 = read.csv(che7, sep = ";",  row.names = 1)
dim(jour212)

aggr(jour212)

plot(1, type = "n", xlim = c(0, 32), ylim = range(jour212, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour212)) {
  ligne = jour212[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "black")
}



# Les profils du troisième jour

### Pour 00h###

che9="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-03_00H_nan-forced.csv"
jour300=read.csv(che9, sep = ";",  row.names = 1)

dim(jour300)
aggr(jour300)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour300, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour300)) {
  ligne = jour300[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "green")
}

### Pour 12h###
che11="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-03_12H_nan-forced.csv"
jour312 = read.csv(che11, sep = ";",  row.names = 1)

aggr(jour312)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour312, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour312)) {
  ligne = jour312[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "brown")
}


# Les profils du 4 ème jour

####Pour 00h####

che13="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-04_00H_nan-forced.csv"
jour400=read.csv(che13, sep = ";",  row.names = 1)
aggr(jour400)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour400, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour400)) {
  ligne = jour400[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "violet")
}


### Pour 12h###

che15="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-04_12H_nan-forced.csv"
jour412 = read.csv(che15, sep = ";",  row.names = 1)


aggr(jour412)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour412, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour412)) {
  ligne = jour412[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "orange")
}


# Les profils du 5 ème jour

### Pour 00h###

che17="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-05_00H_nan-forced.csv"
jour500=read.csv(che17, sep = ";",  row.names = 1)
aggr(jour500)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour500, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour500)) {
  ligne = jour500[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "pink")
}



### Pour 12h###

che19="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-05_12H_nan-forced.csv"
jour512 = read.csv(che19, sep = ";",  row.names = 1)

aggr(jour512)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour512, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour512)) {
  ligne = jour512[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "maroon")
}

# Les profils du 6 ème jour

### Pour 00h###

che21="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-06_00H_nan-forced.csv"
jour600=read.csv(che21, sep = ";",  row.names = 1)

aggr(jour600)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour600, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour600)) {
  ligne = jour600[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "cyan")
}

### Pour 12h###

che23="C:/Users/djafa/Desktop/EtudeStatistique/var_SV_2018-01-06_12H_nan-forced.csv"
jour612 = read.csv(che23, sep = ";",  row.names = 1)

aggr(jour612)
plot(1, type = "n", xlim = c(0, 32), ylim = range(jour612, na.rm = TRUE),
     xlab = "Niveaux de densité", ylab = "Vitesse du son")

# Boucle pour tracer chaque ligne après avoir supprimé les NaN
for (i in 1:nrow(jour612)) {
  ligne = jour612[i, ]
  non_na_indices = which(!is.na(ligne))
  ligne_clean = ligne[non_na_indices]
  densite_clean = densite[non_na_indices]
  lines(densite_clean, ligne_clean,col = "coral")
}










############################## 2- Transformation des profils en données continue##############




##Notre objectif est de travailler sur les profils de longueur maximale pour le premier jour à 12h

# Sélection des lignes avec exactement une donnée manquante et suppression de la dernière colonne

Base2 = jour112[apply(jour112, 1, function(row) sum(is.na(row)) == 1), ]


Base2 = Base2[, -ncol(Base2)]
dim(Base2)

# Représentation des profils de longueur maximale

matplot(t(Base2), type = "l", col = "blue", lty = 1, 
        xlab = "Densité", ylab = "Vitesse")

## Transformation des profils de longueur maximale en données fonctionnelles




# A l'aide de B-spline avec 13 éléments arbitrairement choisi

library(fda)



densite = 0:31


optimal_basis = create.bspline.basis(rangeval = c(0, 31), nbasis = 13)

fdobjvitesse =Data2fd(densite, t(Base2), optimal_basis)


plot(fdobjvitesse, xlab = "Densite", ylab = "Vitesse")

# Recherche du nombre optimal de la base par Validation croisée



compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)# Prédiction pour les données de test
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]# Extraire uniquement les prédictions pour les indices de test valides
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)# Calculer l'erreur
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")# Impression de débogage
  }
  
  return(mean(errors))
}




# Définir une plage de nombres de bases à tester
nbasis_values = seq(5, 50, by = 2)


cv_errors =sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base2))
})

dim(Base2)
optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base2), optimal_basis1)

plot(vitesse_fd_optimal1, xlab = "densité", ylab = "Vitesse")




# Affichage de l'évolution de l'erreur quadratique moyenne
plot(nbasis_values, cv_errors, type = "b", 
     xlab = "Nombre d'éléments de la base B-spline", 
     ylab = "Erreur quadratique moyenne",
     col = "blue", pch = 19)


lines(nbasis_values, cv_errors, col = "blue", lty = 1)


optimal_nbasis_index = which.min(cv_errors)
optimal_nbasis = nbasis_values[optimal_nbasis_index]


points(optimal_nbasis, cv_errors[optimal_nbasis_index], col = "red", pch = 19)
text(optimal_nbasis, cv_errors[optimal_nbasis_index], labels = paste("Optimal:", optimal_nbasis), pos = 3, col = "red")


# A l'aide de la base de fourier avec 13 éléments arbitrarement choisi

matplot(t(Base2), type = "l", col = "blue", lty = 1, 
        xlab = "Densité", ylab = "Célérité")

fourier_basis = create.fourier.basis(rangeval = c(0, 31), nbasis = 13)



vitesse_fd_fourier = Data2fd(densite, t(Base2), fourier_basis)

plot(vitesse_fd_fourier, xlab = "Densité", ylab = "Vitesse")



# Recherche du nombre optimal de la base par Validation croisée


compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.fourier.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)# Prédiction pour les données de test
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]# Extraire uniquement les prédictions pour les indices de test valides
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)# Calculer l'erreur
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")# Impression de débogage
  }
  
  return(mean(errors))
}



# Définir une plage de nombres de bases à tester
nbasis_values2 = seq(5, 32, by = 2)


cv_errors2 =sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base2))
})


optimal_nbasis2 = nbasis_values[which.min(cv_errors2)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis2))


optimal_basis2 = create.fourier.basis(rangeval = c(0, 31), nbasis = optimal_nbasis2)


vitesse_fd_optimal2 = Data2fd(densite, t(Base2), optimal_basis2)

plot(vitesse_fd_optimal2, xlab = "densité", ylab = "Vitesse")









# Affichage de l'évolution de l'erreur quadratique moyenne
plot(nbasis_values2, cv_errors2, type = "b", 
     xlab = "Nombre d'éléments de la base B-spline", 
     ylab = "Erreur quadratique moyenne",
     col = "blue", pch = 19)


lines(nbasis_values2, cv_errors2, col = "blue", lty = 1)


optimal_nbasis_index2 = which.min(cv_errors2)
optimal_nbasis2 = nbasis_values2[optimal_nbasis_index2]


points(optimal_nbasis2, cv_errors2[optimal_nbasis_index2], col = "red", pch = 19)
text(optimal_nbasis2, cv_errors2[optimal_nbasis_index2], labels = paste("Optimal:", optimal_nbasis2), pos = 3, col = "red")














############################## 3- ACP des données fonctionnelles obtenues##############



# ACP fonctionnelle


##Nous rappelons que nos données fonctionnelles obtenues avec B-spline sont dans *vitesse_fd_optimal1*


Temp_ACP = pca.fd(vitesse_fd_optimal1, nharm = 10)#ACPF avec 10 composantes

# Figure (a) : Tracé des harmoniques principales avec pourcentage de variance expliquée
op = par(mfrow = c(1, 1))
plot(Temp_ACP$harmonics[1], xlab = "Densités", ylab = "Valeurs Fct Propre", 
     main = paste("PC1 (", round(variance_expliquee[1], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[2], xlab = "Densités", ylab = "Valeurs Fct Propre", 
     main = paste("PC2 (", round(variance_expliquee[2], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[3], xlab = "Densités", ylab = "Valeurs Fct Propre",  
     main = paste("PC3 (", round(variance_expliquee[3], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[4], xlab = "Densités", ylab = "Valeurs Fct Propre",  
     main = paste("PC4 (", round(variance_expliquee[4], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[5], xlab = "Densitéq", ylab = "Valeurs Fct Propre",  
     main = paste("PC1 (", round(variance_expliquee[5], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[6], xlab = "Densitéq", ylab = "Valeurs Fct Propre",  
     main = paste("PC1 (", round(variance_expliquee[6], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[7], xlab = "Densitéq", ylab = "Valeurs Fct Propre",  
     main = paste("PC1 (", round(variance_expliquee[7], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[8], xlab = "Densitéq", ylab = "Valeurs Fct Propre", 
     main = paste("PC1 (", round(variance_expliquee[8], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[9], xlab = "Densitéq", ylab = "Valeurs Fct Propre",  
     main = paste("PC1 (", round(variance_expliquee[9], 2), "%)", sep = ""))
plot(Temp_ACP$harmonics[10], xlab = "Densitéq", ylab = "Valeurs Fct Propre", 
     main = paste("PC1 (", round(variance_expliquee[10], 2), "%)", sep = ""))


# Figure(B): de l'évolution des 4 premières valeurs propres (scree plot)
eigenvalues = Temp_ACP$values
plot(1:10, eigenvalues[1:10], type = "b", xlab = "Indice j", ylab = "Valeurs propres", col = "red", pch = 19)
lines(1:10, eigenvalues[1:10], col = "red", lty = 1)

## Figure(C): Tracé du graphique de la variance cumulée

variance_expliquee =Temp_ACP$values / sum(Temp_ACP$values) * 100

cumulative_variance = cumsum(explained_variance)

plot(1:10, cumulative_variance[1:10], type = "b", xlab = "Nombre de composantes principales", 
     ylab = "Pourcentage de variance cumulée", 
     col = "blue", pch = 19, ylim = c(0, 100))
lines(1:10, cumulative_variance[1:10], col = "red", lty = 1)

text(1:10, cumulative_variance[1:10], labels = paste0(round(cumulative_variance[1:10], 2), "%"), pos = 3, col = "blue")





# Figure (D) : Tracé des scores des deux premières composantes principales avec les couleurs des clusters

par(mfrow = c(1, 1))

xlim_range = range(Temp_ACP$scores[, 1])

ylim_range = c(-15, 15)
plot(Temp_ACP$scores[, 1], Temp_ACP$scores[, 2], 
     
     pch = 19, 
     cex = 0.5,  
     xlab = "coord. suivant PC1", 
     ylab = "coord. suivant PC2",
     xlim = xlim_range, 
     ylim = ylim_range)


############################## 4-Méthodes de clustering sur espace fini##############





##################4-1 Méthodes par filtrage

###################Extraction des coefficients et des scores###########

coefficients = coef(vitesse_fd_optimal1)# extraction des coefficients
coefficients=t(coefficients)# La matrice transposée des coefficients


scores = Temp_ACP$scores# extraction des scores de l'ACPF
dim(scores)

###################kmean sur les coeff###########




#Choix du nombre de classe par le critère d'instabilité
### L'algo de recherche du nombre par étant couté nous avons fait la recherche sur un échantillon de taille 3000
library(fpc)
set.seed(123)  
indices = sample(1:nrow(coefficients), size = 3000, replace = FALSE)## 
coefficients_sampled = coefficients[indices, ]
dim(coefficients_sampled)
install.packages("fpc")
library(fpc)
set.seed(123)

krange=2:10
res = nselectboot(coefficients_sampled,B=50,clustermethod=kmeansCBI,krange=krange)
res$stabk

res$kopt
plot(krange, res$stabk[-1], type = "b", xlab = "Nombre de clusters (k)", ylab = "Critère d'instabilité")


#Choix du nombre de cluster avec l'inertie intra-classe
wss = numeric(length = 10)  

for (i in 1:10) {
  kmeans_result = kmeans(coefficients, centers = i)
  wss[i] = kmeans_result$tot.withinss
}

plot(1:10, wss, type = "b", xlab = "Nombre de clusters (k)", ylab = "Inertie intra-classe")



# On a déterminé que le nombre optimal de clusters est k = 3
k = 3
set.seed(123)
# Application du K-means sur les coefficients d'expansion avec k=3
kmeans_result = kmeans(coefficients, centers = k)




cluster_assignments = kmeans_result$cluster

# Visualisation des centroïdes des trois clusters
par(mfrow = c(1, 1))  
y_min1 = min(kmeans_result$centers)
y_max1 = max(kmeans_result$centers)

plot(kmeans_result$centers[1, ], type = "l", col = 1,
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min1 - 1, y_max1 + 1))  

lines(kmeans_result$centers[2, ], col = 2)
lines(kmeans_result$centers[3, ], col = 3)

legend("topleft", legend = c("Centroïde 1", "Centroïde 2", "Centroïde 3"),
       col = 1:3, lty = 1, cex = 0.8,
       title = "Centroïdes")





# Visualisation des individus
y_min = min(c(coefficients, kmeans_result$centers))
y_max = max(c(coefficients, kmeans_result$centers))

plot(coefficients[1, ], type = "l", col = cluster_assignments[1],
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min - 1, y_max + 1))  

for (i in 2:nrow(coefficients)) {
  lines(coefficients[i, ], type = "l", col = cluster_assignments[i])
}

for (j in 1:k) {
  lines(kmeans_result$centers[j, ], col = "black", lwd = 2)
}

legend("topleft", legend = paste("Cluster", 1:k),
       col = 1:k, lty = 1, cex = 0.8,
       title = "Clusters")



# Visualisation des trois clusters
par(mfrow = c(1, 3))

for (j in 1:k) {
  
  cluster_indices <- which(cluster_assignments == j)
  
  plot(coefficients[cluster_indices[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
       ylim = c(y_min - 1, y_max + 1))  # Ajuster les valeurs selon vos besoins
  
  for (i in cluster_indices[-1]) {
    lines(coefficients[i, ], type = "l", col = j)
  }
  
  lines(kmeans_result$centers[j, ], col = "black", lwd = 2)
}

par(mfrow = c(1, 1))


# Le nombre d'éléments dans chaque cluster
cluster_counts =table(cluster_assignments)
print(cluster_counts)




########## kmeans sur les scores des composantes#####################

# Choix du nombre de cluster avec l'inertie intra-classe
wss1 = numeric(length = 10)  

for (i in 1:10) {
  kmeans_result1 = kmeans(scores, centers = i)
  wss1[i] = kmeans_result1$tot.withinss
}

plot(1:10, wss1, type = "b", xlab = "Nombre de clusters (k)", ylab = "Inertie intra-classe")


#Choix du nombre de classe par le critère d'instabilité
set.seed(123)  
indices2 = sample(1:nrow(scores), size = 3000, replace = FALSE)
scores_sampled = scores[indices2, ]
install.packages("fpc")
library(fpc)
set.seed(123)

krange=2:10
res2 = nselectboot(scores_sampled ,B=50,clustermethod=kmeansCBI,krange=krange)
res2$stabk

res2$kopt
plot(krange, res2$stabk[-1], type = "b", xlab = "Nombre de clusters (k)", ylab = "Critère d'instabilité")







# On a déterminé que le nombre optimal de clusters est k = 3
k = 3
set.seed(123)
# Application du K-means sur les scores avec k=3
kmeans_result2 = kmeans(scores, centers = k)


cluster_assignments2 = kmeans_result2$cluster








# Visualisation des trois clusters
par(mfrow = c(1, 3))

for (j in 1:k) {
  
  cluster_indices1 <- which(cluster_assignments2 == j)
  
  plot(scores[cluster_indices[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des scores", ylab = "Valeur des scores",
       ylim = c(y_min - 1, y_max + 1))  # Ajuster les valeurs selon vos besoins
  
  for (i in cluster_indices1[-1]) {
    lines(scores[i, ], type = "l", col = j)
  }
  
  lines(kmeans_result2$centers[j, ], col = "black", lwd = 2)
}

par(mfrow = c(1, 1))


# Représentation des individus sur les plans de l'acp
par(mfrow = c(1, 1))

xlim_range = range(Temp_ACP$scores[, 1])

ylim_range = c(-15, 15)
plot(Temp_ACP$scores[, 1], Temp_ACP$scores[, 2], 
     col = cluster_assignments2, 
     pch = 19, 
     cex = 0.5,  
     xlab = "coord. suivant PC1", 
     ylab = "coord. suivant PC2",
     xlim = xlim_range, 
     ylim = ylim_range)

legend("bottomright", legend = paste("Cluster", 1:k),
       col = 1:k, pch = 19, 
       title = "Clusters")
# Le nombre d'éléments dans chaque cluster
cluster_counts2 =table(cluster_assignments2)
print(cluster_counts2)


################ CAH sur les coeff avec critère de ward###############

library(cluster)

distance_matrix = dist(coefficients)



cah_result = hclust(distance_matrix, method = "ward.D2")
summary(cah_result)

plot(cah_result)
k = 3
rect.hclust(cah_result, k = k, border = 2:4)
# Découpage en trois clusters
k = 3
clusters = cutree(cah_result, k = k)



# Visualisation des moyennes des clusters

centers = t(sapply(unique(clusters), function(cluster_id) {
  colMeans(coefficients[clusters == cluster_id, , drop = FALSE])
}))

par(mfrow = c(1, 1))
y_min1 = min(centers)
y_max1 = max(centers)

plot(centers[1, ], type = "l", col = 1,
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min1 - 1, y_max1 + 1))

for (i in 2:nrow(centers)) {
  lines(centers[i, ], col = i)
}

legend("topleft", legend = paste("Centroïde", 1:nrow(centers)),
       col = 1:nrow(centers), lty = 1, cex = 0.8,
       title = "Centroïdes")

# Visualisation des individus par cluster
y_min = min(c(coefficients, centers))
y_max = max(c(coefficients, centers))

plot(coefficients[1, ], type = "l", col = clusters[1],
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min - 1, y_max + 1))

for (i in 2:nrow(coefficients)) {
  lines(coefficients[i, ], type = "l", col = clusters[i])
}

for (j in 1:k) {
  lines(centers[j, ], col = "black", lwd = 2)
}

legend("topleft", legend = paste("Cluster", 1:k),
       col = 1:k, lty = 1, cex = 0.8,
       title = "Clusters")

# Visualisation des clusters individuels
par(mfrow = c(1, k))

for (j in 1:k) {
  cluster_indices = which(clusters == j)
  
  plot(coefficients[cluster_indices[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
       ylim = c(y_min - 1, y_max + 1))
  
  for (i in cluster_indices[-1]) {
    lines(coefficients[i, ], type = "l", col = j)
  }
  
  lines(centers[j, ], col = "black", lwd = 2)
}

par(mfrow = c(1, 1))



# Compte du nombre d'éléments dans chaque cluster
cluster_counts = table(clusters)
cat("Nombre d'éléments dans chaque cluster:\n")
print(cluster_counts)


















################ De la CAH avec critère du saut maximal##############################

cah_result3 = hclust(distance_matrix, method = "complete")
plot(cah_result3)

k = 3
rect.hclust(cah_result3, k = k, border = 2:4)  

# Découpage en trois clusters
clusters3 = cutree(cah_result3, k = k)


# Visualisation des moyennes des clusters
centers3 = t(sapply(unique(clusters3), function(cluster_id) {
  colMeans(coefficients[clusters3 == cluster_id, , drop = FALSE])
}))

par(mfrow = c(1, 1))
y_min1 = min(centers3)
y_max1 = max(centers3)

plot(centers3[1, ], type = "l", col = 1,
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min1 - 1, y_max1 + 1))

for (i in 2:nrow(centers3)) {
  lines(centers3[i, ], col = i)
}

legend("topleft", legend = paste("Centroïde", 1:nrow(centers3)),
       col = 1:nrow(centers3), lty = 1, cex = 0.8,
       title = "Centroïdes")

# Visualisation des individus par cluster
y_min = min(c(coefficients, centers3))
y_max = max(c(coefficients, centers3))

plot(coefficients[1, ], type = "l", col = clusters3[1],
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min - 1, y_max + 1))

for (i in 2:nrow(coefficients)) {
  lines(coefficients[i, ], type = "l", col = clusters3[i])
}

for (j in 1:k) {
  lines(centers3[j, ], col = "black", lwd = 2)
}

legend("topleft", legend = paste("Cluster", 1:k),
       col = 1:k, lty = 1, cex = 0.8,
       title = "Clusters")

# Visualisation des clusters individuels
par(mfrow = c(1, k))

for (j in 1:k) {
  cluster_indices = which(clusters3 == j)
  
  plot(coefficients[cluster_indices[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
       ylim = c(y_min - 1, y_max + 1))
  
  for (i in cluster_indices[-1]) {
    lines(coefficients[i, ], type = "l", col = j)
  }
  
  lines(centers3[j, ], col = "black", lwd = 2)
}

par(mfrow = c(1, 1))

# Compte du nombre d'éléments dans chaque cluster
cluster_counts3 = table(clusters3)
cat("Nombre d'éléments dans chaque cluster:\n")
print(cluster_counts3)







################## De la CAH avec saut minimal#########################

cah_result2 = hclust(distance_matrix, method = "single")
plot(cah_result2)
k = 3
rect.hclust(cah_result2, k = k, border = 2:4)




##################### CAH sur les scores Avec ward##########################






distance_matrix1 = dist(scores)


cah_result4 = hclust(distance_matrix1, method = "ward.D2")


plot(cah_result4)


k = 2
rect.hclust(cah_result4, k = k, border = 2:4)


clusters4 = cutree(cah_result4, k = k)



# Visualisation des clusters
par(mfrow = c(1, k))

y_min = min(scores)
y_max = max(scores)

for (j in 1:k) {
  cluster_indices <- which(clusters4 == j)
  
  plot(scores[cluster_indices[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des scores", ylab = "Valeur des scores",
       ylim = c(y_min - 1, y_max + 1))
  
  for (i in cluster_indices[-1]) {
    lines(scores[i, ], type = "l", col = j)
  }
}

par(mfrow = c(1, 1))

# Représentation des individus sur les deux premières composantes principales de l'ACP
xlim_range = range(Temp_ACP$scores[, 1])
ylim_range = c(-15, 15)

plot(Temp_ACP$scores[, 1], Temp_ACP$scores[, 2], 
     col = clusters4, 
     pch = 19, 
     cex = 0.5,  
     xlab = "coord. suivant PC1", 
     ylab = "coord. suivant PC2",
     xlim = xlim_range, 
     ylim = ylim_range)

legend("bottomright", legend = paste("Cluster", 1:k),
       col = 1:k, pch = 19, 
       title = "Clusters")



# Compte du nombre d'éléments dans chaque cluster
cluster_counts4 = table(clusters4)
cat("Nombre d'éléments dans chaque cluster:\n")
print(cluster_counts4)






#######De la classification CAH avec la méthode de de saut maximal##################

cah_result6 = hclust(distance_matrix1, method = "complete")
plot(cah_result6)
k = 2
rect.hclust(cah_result6, k = k, border = 2:4)



clusters6 = cutree(cah_result6, k = k)



# Visualisation des clusters
par(mfrow = c(1, k))

y_min = min(scores)
y_max = max(scores)

for (j in 1:k) {
  cluster_indices <- which(clusters4 == j)
  
  plot(scores[cluster_indices[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des scores", ylab = "Valeur des scores",
       ylim = c(y_min - 1, y_max + 1))
  
  for (i in cluster_indices[-1]) {
    lines(scores[i, ], type = "l", col = j)
  }
}

par(mfrow = c(1, 1))

# Représentation des individus sur les deux premières composantes principales de l'ACP
xlim_range = range(Temp_ACP$scores[, 1])
ylim_range = c(-15, 15)

plot(Temp_ACP$scores[, 1], Temp_ACP$scores[, 2], 
     col = clusters4, 
     pch = 19, 
     cex = 0.5,  
     xlab = "coord. suivant PC1", 
     ylab = "coord. suivant PC2",
     xlim = xlim_range, 
     ylim = ylim_range)

legend("bottomright", legend = paste("Cluster", 1:k),
       col = 1:k, pch = 19, 
       title = "Clusters")



# Compte du nombre d'éléments dans chaque cluster
cluster_counts6 = table(clusters6)
cat("Nombre d'éléments dans chaque cluster:\n")
print(cluster_counts6)




################ De la CAH avec saut minimal#######################

cah_result7 = hclust(distance_matrix1, method = "single")
plot(cah_result7)
k = 3
rect.hclust(cah_result7, k = k, border = 2:4)


##############Algorithme PAM sur les coefficients d'expansion###############

#Choix du nombre de classe par le critère de la statistique de gap


set.seed(123)
gap_stat = clusGap(
  coefficients_sampled,
  FUNcluster = pam,
  K.max = 10,    
  B = 50        
)

fviz_gap_stat(gap_stat)



#Choix du nombre de classe par le critère d'instabilité


install.packages("fpc")
library(fpc)
set.seed(123)

krange1=2:10
res1 = nselectboot(coefficients_sampled,B=50,clustermethod=claraCBI,krange=krange1)
res1$stabk

res1$kopt
plot(krange1, res1$stabk[-1], type = "b", xlab = "Nombre de clusters (k)", ylab = "Critère d'instabilité")


#Application de PAM avec k=3
set.seed(123)
pam_result_coeff = pam(coefficients, 3)




pam_cluster_assignments_coeff = pam_result_coeff$clustering
# Visualisation des medoids des trois clusters
par(mfrow = c(1, 1))  
y_min_pam = min(pam_result_coeff$medoids)
y_max_pam = max(pam_result_coeff$medoids)

plot(pam_result_coeff$medoids[1, ], type = "l", col = 1,
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min_pam - 1, y_max_pam + 1))  


lines(pam_result_coeff$medoids[2, ], col = 2)
lines(pam_result_coeff$medoids[3, ], col = 3)

legend("topleft", legend = c("Medoid 1", "Medoid 2", "Medoid 3"),
       col = 1:3, lty = 1, cex = 0.8,
       title = "Medoids")

# Visualisation des individus
y_min_ind_pam = min(c(coefficients, pam_result_coeff$medoids))
y_max_ind_pam = max(c(coefficients, pam_result_coeff$medoids))

plot(coefficients[1, ], type = "l", col = pam_cluster_assignments_coeff[1],
     xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
     ylim = c(y_min_ind_pam - 1, y_max_ind_pam + 1))  

for (i in 2:nrow(coefficients)) {
  lines(coefficients[i, ], type = "l", col = pam_cluster_assignments_coeff[i])
}

for (j in 1:3) {
  lines(pam_result_coeff$medoids[j, ], col = "black", lwd = 2)
}

legend("topleft", legend = paste("Cluster", 1:3),
       col = 1:3, lty = 1, cex = 0.8,
       title = "Clusters")


# Visualisation des trois clusters individuellement
par(mfrow = c(1, 3))

for (j in 1:3) {
  
  cluster_indices_pam <- which(pam_cluster_assignments_coeff == j)
  
  plot(coefficients[cluster_indices_pam[1], ], type = "l", col = j,
       main = paste("Cluster", j),
       xlab = "Indexes des coefficients", ylab = "Valeur du coefficient",
       ylim = c(y_min_ind_pam - 1, y_max_ind_pam + 1))  # Ajuster les valeurs selon vos besoins
  
  for (i in cluster_indices_pam[-1]) {
    lines(coefficients[i, ], type = "l", col = j)
  }
  
  lines(pam_result_coeff$medoids[j, ], col = "black", lwd = 2)
}

par(mfrow = c(1, 1))


# Le nombre d'éléments dans chaque cluster
pam_cluster_counts = table(pam_cluster_assignments_coeff)
print(pam_cluster_counts)




########Algorithme PAM sur les scores de composantes principales############



#Choix du nombre de classe par le critère de la statistique de gap
set.seed(123)
gap_stat2 = clusGap(
  scores_sampled,
  FUNcluster = pam,
  K.max = 10,    
  B = 50        
)


fviz_gap_stat(gap_stat2)

#Choix du nombre de classe par le critère d'instabilité


install.packages("fpc")
library(fpc)
set.seed(123)

krange2=2:10
res2 = nselectboot(scores_sampled,B=50,clustermethod=claraCBI,krange=krange2)
res2$stabk

res2$kopt
plot(krange2, res2$stabk[-1], type = "b", xlab = "Nombre de clusters (k)", ylab = "Critère d'instabilité")

# Application de PAM avec k=3
set.seed(123)

pam_result_score= pam(scores, 3)


################ L'algoritme EM sur les coefficients d'expansions##############


library(mclust)

#Classe optimale par icl

G_range = 1:10

icl_values =numeric(length(G_range))
for (G in G_range) {
  
  model = Mclust(coefficients, G = G)
  
  icl_values[G] = model$icl
}

optimal_G = G_range[which.max(icl_values)]


cat("Nombre optimal de clusters selon le critère ICL:", optimal_G, "\n")

plot(G_range, icl_values, type = "b", xlab = "Nombre de clusters", ylab = "Critère ICL")


#Application de l'algo EM avec G=9
model = Mclust(coefficients, G = 9)







############### L'algoritme EM sur les scores de composantes principales########
#Classe optimale par icl


G_range1 = 1:10

icl_values1 =numeric(length(G_range1))

for (G in G_range1) {
  
  model1 = Mclust(scores, G = G)
  
  icl_values1[G] = model1$icl
}

optimal_G1 = G_range1[which.max(icl_values1)]


cat("Nombre optimal de clusters selon le critère ICL:", optimal_G1, "\n")

plot(G_range1, icl_values1, type = "b", xlab = "Nombre de clusters", ylab = "Critère ICL")


#Application de l'algo EM avec G=10

model1 = Mclust(scores, G = 10)








############### Calcul des indices de qualité des algo kmeans,CAH et PAM###################

library(cluster)  
install.packages("clusterSim")
library(clusterSim)

## Calcul de l'indice de silhouette et de l'indice de Davies-Bouldin pour le kmeans sur les coeff
# Indice de silhouette
silhouette_coeff = silhouette(cluster_assignments, dist(coefficients))
avg_silhouettecoeff = mean(silhouette_coeff[, 3])
cat("L'indice de silhouette moyen est :", avg_silhouettecoeff, "\n")

# Indice de Davies-Bouldin

db_indexcoeff = index.DB(coefficients, cluster_assignments, d=dist(coefficients))$DB


cat("L'indice de Davies-Bouldin est :", db_indexcoeff, "\n")



## Calcul de l'indice de silhouette et de l'indice de Davies-Bouldin pour le kmeans sur les scores
# Indice de silhouette

silhouette_scores = silhouette(cluster_assignments2, dist(scores))


avg_silhouette = mean(silhouette_scores[, 3])
cat("L'indice de silhouette moyen est :", avg_silhouette, "\n")

# Indice de Davies-Bouldin

db_indexscores = index.DB(scores, cluster_assignments2, d=dist(scores))$DB


cat("L'indice de Davies-Bouldin est :", db_indexscores, "\n")



## Calcul de l'indice de silhouette et de l'indice de Davies-Bouldin sur CAH avec ward sur coeff 

# Indice de silhouette
silhouette_cah_ward_coeff = silhouette(clusters, dist(coefficients))
silhouette_index_cah_ward_coeff = mean(silhouette_cah_ward_coeff[, 3])
cat("Indice de silhouette moyen :", silhouette_index_cah_ward_coeff, "\n")

# Indice de Davies-Bouldin

db_index_cah_ward_coeff = index.DB(coefficients, clusters, centrotypes = "centroids")$DB
cat("Indice de Davies-Bouldin :", db_index_cah_ward_coeff, "\n")

## Calcul de l'indice de silhouette et de l'indice de Davies-Bouldin sur CAH avec saut maximum sur coeff 


# Indice de silhouette
silhouette_cah_complete = silhouette(clusters3, dist(coefficients))
silhouette_index_cah_complete = mean(silhouette_cah_complete[, 3])
cat("Indice de silhouette moyen :", silhouette_index_cah_complete, "\n")





# Indice de Davies-Bouldin

db_index_cah_complete = index.DB(coefficients, clusters3, centrotypes = "centroids")$DB
cat("Indice de Davies-Bouldin :", db_index_cah_complete, "\n")


## Calcul de l'indice de silhouette et de l'indice de Davies-Bouldin sur CAH avec ward sur les scores

# Indice de silhouette

silhouette_cah_ward_scores = silhouette(clusters4, dist(scores))
silhouette_index_cah_ward_scores = mean(silhouette_cah_ward_scores[, 3])
cat("Indice de silhouette moyen :", silhouette_index_cah_ward_scores, "\n")

# Indice de Davies-Bouldin


db_index_cah_ward_scores = index.DB(scores, clusters4, centrotypes = "centroids")$DB
cat("Indice de Davies-Bouldin :", db_index_cah_ward_scores, "\n")



## Calcul de l'indice de silhouette et de l'indice de Davies-Bouldin sur CAH avec saut maximum sur les scores



# Indice de silhouette

silhouette_cah_saut_scores = silhouette(clusters6, dist(scores))
silhouette_index_cah_saut_scores = mean(silhouette_cah_saut_scores[, 3])
cat("Indice de silhouette moyen :", silhouette_index_cah_saut_scores, "\n")

# Indice de Davies-Bouldin

db_index_cah_saut_scores = index.DB(scores, clusters6, centrotypes = "centroids")$DB
cat("Indice de Davies-Bouldin :", db_index_cah_saut_scores, "\n")





##################4-2 Méthodes Adaptatives#######################



############### L'algoritme funFEM#########
install.packages("funFEM")
library(funFEM)


#Choix du meilleur modèle


Res33=funFEM(fd = vitesse_fd_optimal1, K = 2:10, model = "all",crit="icl", init='kmeans', lambda=0, disp=TRUE)


#Choix du nombre de classe avec le meileur modèle AkjB 
Res34=funFEM(fd = vitesse_fd_optimal1, K = 2:10, model = "AkjB",crit="icl", init='kmeans', lambda=0, disp=TRUE)

icl_values34 = Res34$allCriterions$icl

num_clusters34 = 2:10
par(mfrow=c(1,1))
plot(num_clusters34, icl_values34, type = "b", xlab = "Nombre de clusters (K)", ylab = "ICL", xaxt = "n")
axis(1, at = num_clusters34, labels = as.character(num_clusters34))

# Représentation des profils de chaque groupe et la courbe moyenne de chaque groupe
par(mfrow=c(1,2))
plot(vitesse_fd_optimal1)
lines(vitesse_fd_optimal1,col=Res34$cls,lwd=2,lty=1)
fdmeans = vitesse_fd_optimal1
fdmeans$coefs = t(Res34$prms$my)
plot(fdmeans); lines(fdmeans,col=1:max(Res34$cls),lwd=2)



# Visualization in the discriminative subspace (projected scores)
par(mfrow=c(1,1))
plot(t(vitesse_fd_optimal1$coefs) %*% Res34$U,col=Res34$cls,pch=19,main="Discriminative space")


################# L'algoritme Funclustering#################


pkgbuild::check_build_tools(debug = TRUE)



install.packages("Funclustering", repos="http://R-Forge.R-project.org")


library(devtools)
install_github("modal-inria/Funclustering")


library(Funclustering)
library(fda)

#Choix du nombre de classe
icl_values <- numeric()
execution_times <- numeric()  


for (K in 2:10) {
  start_time <- proc.time()  
  resF1 <- funclust(vitesse_fd_optimal1, K = K)
  end_time <- proc.time()  
  
  
  execution_time <- end_time - start_time
  
  icl_values <- c(icl_values, resF1$icl)
  execution_times <- c(execution_times, execution_time[3])  
  
  print(paste("Nombre de clusters:", K, "ICL:", resF1$icl, "Temps d'exécution (sec):", execution_time[3]))
}


plot(2:10, icl_values, type = "b", xlab = "Nombre de clusters", ylab = "ICL")


print(execution_times)

#Application de l'algo funclust avec k=2

res99=funclust(fd, K = 2)

summary(res99)

# Représentation des profils de chaque groupe et la courbe moyenne de chaque groupe
par(mfrow=c(1,1))
plot(fd,pch=19,cex=0.5,xlab="Densite",ylab="Vitesse")
lines(fd,col=res99$cls,lwd=2,lty=1)

legend("topleft", legend = paste("Cluster", 1:max(res99$cls)),
       col = 1:max(res99$cls), lty = 1, lwd = 2, cex = 0.8,
       title = "Clusters")




############################## 5-Méthodes de clustering sur espace infini##############





######## CLUSTERING AVEC LE PACKAGE Clustering avec le package fda.usc######

install.packages("fda.usc")

library(fda.usc)

library(fda)

out.fd1=kmeans.fd(Base2,ncl=3,draw=TRUE)




############################## 5-Carte topographique : répartition des pixels par cluster dans la zone du Golfe du Lion et comparaison de quelques méthodes de clustering##############






####### Chargement des cordonnées
cordonnees ="C:/Users/djafa/Desktop/EtudeStatistique/coords_LON+3+6-LAT+40+42.csv"
corgeo = read.csv(cordonnees, sep = ";", header = TRUE)
head(corgeo)

#Je rappelle que Base2 contient les profils de longueur maximale


pixels_communs = Base2$pixel #Extraire la colonne 'pixel' de la base de données Base2


corgeo_filtre = corgeo[corgeo$pixel %in% pixels_communs, ] # Filtrer le tableau 'corgeo' pour ne garder que les lignes dont la colonne 'pixel' correspond à un pixel présent dans 'pixels_communs'


head(corgeo_filtre)
dim(corgeo_filtre)

colnames(corgeo_filtre)[colnames(corgeo_filtre) == "pixel"] = "" # Renommer la colonne 'pixel' dans 'corgeo_filtre' en supprimant son nom






##### Représentation dans le plan des pixels pour le kmeans sur les coeff

library(ggplot2)
set.seed(123)


num_clusters = length(unique(cluster_assignments))


colors = c("black", "brown", "green")


corgeo_filtre$color = colors[as.factor(cluster_assignments)]


plot(corgeo_filtre$lon, corgeo_filtre$lat, 
     col = corgeo_filtre$color, 
     pch = 16, 
     cex = 0.3,  
     xlab = "Longitude", 
     ylab = "Latitude")


legend("topleft", legend = paste("Cluster", 1:num_clusters), 
       col = colors, pch = 16, 
       title = "Clusters")



##### Représentation dans le plan des pixels pour le kmeans sur les scores



corgeo_filtre$color = colors[as.factor(cluster_assignments2)]


plot(corgeo_filtre$lon, corgeo_filtre$lat, 
     col = corgeo_filtre$color, 
     pch = 16, 
     cex = 0.1,  
     xlab = "Longitude", 
     ylab = "Latitude")


legend("topleft", legend = paste("Cluster", 1:num_clusters), 
       col = colors, pch = 16, 
       title = "Clusters")


##### Représentation dans le plan des pixels pour CAH sur les coeff avec critère de ward



corgeo_filtre$color = colors[as.factor(cluster_assignments2)]


plot(corgeo_filtre$lon, corgeo_filtre$lat, 
     col = corgeo_filtre$color, 
     pch = 16, 
     cex = 0.1,  
     xlab = "Longitude", 
     ylab = "Latitude")


legend("topleft", legend = paste("Cluster", 1:num_clusters), 
       col = colors, pch = 16, 
       title = "Clusters")




##### Représentation dans le plan des pixels pour CAH sur les coeff avec critère de saut maximum



corgeo_filtre$color = colors[as.factor(cluster_assignments2)]


plot(corgeo_filtre$lon, corgeo_filtre$lat, 
     col = corgeo_filtre$color, 
     pch = 16, 
     cex = 0.1,  
     xlab = "Longitude", 
     ylab = "Latitude")


legend("topleft", legend = paste("Cluster", 1:num_clusters), 
       col = colors, pch = 16, 
       title = "Clusters")






##### Représentation dans le plan des pixels pour CAH sur les coeff avec critère de saut mimimum



corgeo_filtre$color = colors[as.factor(cluster_assignments2)]


plot(corgeo_filtre$lon, corgeo_filtre$lat, 
     col = corgeo_filtre$color, 
     pch = 16, 
     cex = 0.1,  
     xlab = "Longitude", 
     ylab = "Latitude")


legend("topleft", legend = paste("Cluster", 1:num_clusters), 
       col = colors, pch = 16, 
       title = "Clusters")












############################# Les travaux sur les profils de longueurs différentes##########



####Ici notre objectif est de transformer tous en utilsant le nombre optimal de base 
####obtenu sur les profils de longueur maximale

#########################Les fichiers CSV

###PRENIER JOUR




##00H


### Nombre de profils dans chaque groupe




na_counts0 = apply(jour100, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0


###Recherche du nombre optimal de la base sur les profils de longueur maximale par validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}

##Rcherche des profils de longueur maximale

Base0 = jour100[apply(jour100, 1, function(row) sum(is.na(row)) == 1), ]


Base0 = Base0[, -ncol(Base0)]

nbasis_values = seq(6, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base0))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 27 comme nombre optimal


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")








# Boucle pour créer Base1 à Base28 car nous avons 28 groupes car nous avons 28 groupes
bases = list()


for (i in 1:28) {
  
  base_temp = jour100[apply(jour100, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation des 28 matrices des coefficients
coefficients_concatJ_1_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_1_00h)) {
    coefficients_concatJ_1_00h = coefficients_temp  
  } else {
    coefficients_concatJ_1_00h = rbind(coefficients_concatJ_1_00h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_1_00h.csv"


write.csv2(coefficients_concatJ_1_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_1_00h)


dim(coefficients_concatJ_1_00h)


### Variances des coeffs de la matrice concatenée



variances_J_1_00h = apply(coefficients_concatJ_1_00h[, 1:27], 2, var)


print(variances_J_1_00h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_1_00h.csv"


write.csv2(variances_J_1_00h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_1_00h)


plot(1:length(variances_J_1_00h), variances_J_1_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_1_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels












##12H

### Nombre de profils dans chaque groupe




na_counts0 = apply(jour112, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0






###Recherche du nombre optimal de la base sur les profils de longueur maximale par validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}

##Rcherche des profils de longueur maximale

Base2 = jour100[apply(jour112, 1, function(row) sum(is.na(row)) == 1), ]


Base2 = Base2[, -ncol(Base2)]

nbasis_values = seq(6, 30, by = 1)### Nous avons cherché le nombre optimal sur les nombre pairs et impairs


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base2))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26 comme nobre optmal


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base2), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")





#### Boucle pour créer les groupes de profils Base1 à  Base28 car nous avons 28 groupes###


bases = list() # Initialiser une liste pour stocker les bases de données


for (i in 1:28) {
  
  base_temp = jour112[apply(jour112, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}



####Tranformation des 28 groupes à l'aide de 26 éléments de la base B-spline en données fonctionnelles et extraction des coeffs
library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation des matrices contenant les coeffs de chaque groupe de profils
coefficients_concatJ_1_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_1_12h)) {
    coefficients_concatJ_1_12h = coefficients_temp  
  } else {
    coefficients_concatJ_1_12h = rbind(coefficients_concatJ_1_12h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_1_12h.csv"


write.csv2(coefficients_concatJ_1_12h, file = file_path, row.names = TRUE)#,sep = ";"

dim(coefficients_concatJ_1_12h)

### Variances des coeffs de la matrice concatenée

variances_J_1_12h = apply(coefficients_concatJ_1_12h[, 1:26], 2, var)


print(variances_J_1_12h)

file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_1_12h.csv"


write.csv2(variances_J_1_12h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances)


plot(1:length(variances), variances, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels














###DEUXIEME JOUR

##00H







### Nombre de profils dans chaque groupe



na_counts0 = apply(jour200, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###Recherche du nombre optimal de la base sur les profils de longueur maximale par validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}


##Rcherche des profils de longueur maximale

Base3 = jour200[apply(jour200, 1, function(row) sum(is.na(row)) == 1), ]


Base3 = Base3[, -ncol(Base3)]

nbasis_values = seq(6, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base3))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base3), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")






# Boucle pour créer Base1 à Base28


bases = list() # Initialiser une liste pour stocker les groupes de données


for (i in 1:28) {
  
  base_temp = jour100[apply(jour200, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_2_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_2_00h)) {
    coefficients_concatJ_2_00h = coefficients_temp  
  } else {
    coefficients_concatJ_2_00h = rbind(coefficients_concatJ_2_00h, coefficients_temp)  
  }
}
### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_2_00h.csv"


write.csv2(coefficients_concatJ_2_00h, file = file_path, row.names = TRUE)#,sep = ";"

dim(coefficients_concatJ_2_00h)




### Variances des coeffs de la matrice concatenée

variances_J_2_00h = apply(coefficients_concatJ_2_00h[, 1:26], 2, var)


print(variances_J_2_00h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_2_00h.csv"


write.csv2(variances_J_2_00h, file = file_path, row.names = TRUE)


### Représentation des varianves
bspl4_names = names(variances_J_2_00h)


plot(1:length(variances_J_2_00h), variances_J_2_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_2_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels


##12H


### Nombre de profils dans chaque groupe




na_counts0 = apply(jour212, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0





###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}


##Rcherche des profils de longueur maximale

Base7 = jour100[apply(jour212, 1, function(row) sum(is.na(row)) == 1), ]


Base7 = Base7[, -ncol(Base7)]
nbasis_values = seq(6, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base7))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 27 


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")








# Boucle pour créer Base1 à Base28

bases = list()


for (i in 1:28) {
  
  base_temp = jour212[apply(jour212, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_2_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_2_12h)) {
    coefficients_concatJ_2_12h = coefficients_temp  
  } else {
    coefficients_concatJ_2_12h = rbind(coefficients_concatJ_2_12h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_2_12h.csv"


write.csv2(coefficients_concatJ_2_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_2_12h)


head(coefficients_concatJ_2_12h)

### Variances des coeffs de la matrice concatenée

variances_J_2_12h = apply(coefficients_concatJ_2_12h[, 1:26], 2, var)


print(variances_J_2_12h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_2_12h.csv"


write.csv2(variances_J_2_12h, file = file_path, row.names = TRUE)


### Représentation des varianves


bspl4_names = names(variances_J_2_12h)


plot(1:length(variances_J_2_12h), variances_J_2_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_2_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels





###TROISIEME JOUR

##00H


### Nombre de profils dans chaque groupe



na_counts0 = apply(jour300, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0



###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}

##Rcherche des profils de longueur maximale

Base9 = jour300[apply(jour300, 1, function(row) sum(is.na(row)) == 1), ]


Base9 = Base9[, -ncol(Base9)]

nbasis_values = seq(6, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base9))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")


 







# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour300[apply(jour300, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_3_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_3_00h)) {
    coefficients_concatJ_3_00h = coefficients_temp  
  } else {
    coefficients_concatJ_3_00h = rbind(coefficients_concatJ_3_00h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_3_00h.csv"


write.csv2(coefficients_concatJ_3_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_3_00h)


dim(coefficients_concatJ_3_00h)

head(coefficients_concatJ_3_00h)

### Variances des coeffs de la matrice concatenée

variances_J_3_00h = apply(coefficients_concatJ_3_00h[, 1:26], 2, var)


print(variances_J_3_00h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_3_00h.csv"


write.csv2(variances_J_1_00h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_3_00h)


plot(1:length(variances_J_3_00h), variances_J_3_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_3_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels


##12H


### Nombre de profils dans chaque groupe




na_counts0 = apply(jour312, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0








###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}

##Rcherche des profils de longueur maximale

Base11 = jour312[apply(jour312, 1, function(row) sum(is.na(row)) == 1), ]


Base11 = Base11[, -ncol(Base11)]

nbasis_values = seq(6, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base11))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")






# Boucle pour créer Base1 à Base28


bases = list()


for (i in 1:28) {
  
  base_temp = jour312[apply(jour312, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_3_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_3_12h)) {
    coefficients_concatJ_3_12h = coefficients_temp  
  } else {
    coefficients_concatJ_3_12h = rbind(coefficients_concatJ_3_12h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_3_12h.csv"


write.csv2(coefficients_concatJ_3_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_3_12h)

head(coefficients_concatJ_3_12h)


### Variances des coeffs de la matrice concatenée
variances_J_3_12h = apply(coefficients_concatJ_3_12h[, 1:26], 2, var)


print(variances_J_3_12h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_3_12h.csv"


write.csv2(variances_J_3_12h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_3_12h)


plot(1:length(variances_J_3_12h), variances_J_3_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_3_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels








###QUATRIEME JOUR

##00H



### Nombre de profils dans chaque groupe





na_counts0 = apply(jour400, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}


##Rcherche des profils de longueur maximale

Base13 = jour100[apply(jour400, 1, function(row) sum(is.na(row)) == 1), ]


Base13 = Base13[, -ncol(Base13)]

nbasis_values = seq(6, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base13))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")








# Boucle pour créer Base1 à Base28
bases = list()


for (i in 1:28) {
  
  base_temp = jour400[apply(jour400, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_4_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_4_00h)) {
    coefficients_concatJ_4_00h = coefficients_temp  
  } else {
    coefficients_concatJ_4_00h = rbind(coefficients_concatJ_4_00h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_4_00h.csv"


write.csv2(coefficients_concatJ_4_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_4_00h)


dim(coefficients_concatJ_4_00h)

head(coefficients_concatJ_4_00h)

### Variances des coeffs de la matrice concatenée

variances_J_4_00h = apply(coefficients_concatJ_4_00h[, 1:26], 2, var)


print(variances_J_4_00h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_4_00h.csv"


write.csv2(variances_J_4_00h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_4_00h)


plot(1:length(variances_J_4_00h), variances_J_4_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_4_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels













##12H


### Nombre de profils dans chaque groupe





na_counts0 = apply(jour412, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}


##Rcherche des profils de longueur maximale

Base15 = jour412[apply(jour412, 1, function(row) sum(is.na(row)) == 1), ]


Base15 = Base15[, -ncol(Base15)]
nbasis_values = seq(5, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base15))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis)) ##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")








# Boucle pour créer Base1 à Base28
bases = list()


for (i in 1:28) {
  
  base_temp = jour412[apply(jour412, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_4_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_4_12h)) {
    coefficients_concatJ_4_12h = coefficients_temp  
  } else {
    coefficients_concatJ_4_12h = rbind(coefficients_concatJ_4_12h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_4_12h.csv"


write.csv2(coefficients_concatJ_4_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_4_12h)

head(coefficients_concatJ_4_12h)


### Variances des coeffs de la matrice concatenée
variances_J_4_12h = apply(coefficients_concatJ_4_12h[, 1:26], 2, var)


print(variances_J_4_12h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_4_12h.csv"


write.csv2(variances_J_4_12h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_3_12h)


plot(1:length(variances_J_3_12h), variances_J_3_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_3_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



###cinquieme JOUR

##00H

### Nombre de profils dans chaque groupe





na_counts0 = apply(jour500, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}


##Rcherche des profils de longueur maximale

Base17 = jour500[apply(jour500, 1, function(row) sum(is.na(row)) == 1), ]


Base17 = Base17[, -ncol(Base17)]

nbasis_values = seq(5, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base17))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26

optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")








# Boucle pour créer Base1 à Base29
bases = list()


for (i in 1:29) {
  
  base_temp = jour500[apply(jour500, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_5_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_5_00h)) {
    coefficients_concatJ_5_00h = coefficients_temp  
  } else {
    coefficients_concatJ_5_00h = rbind(coefficients_concatJ_5_00h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_5_00h.csv"


write.csv2(coefficients_concatJ_5_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_5_00h)


dim(coefficients_concatJ_5_00h)

head(coefficients_concatJ_5_00h)

### Variances des coeffs de la matrice concatenée

variances_J_5_00h = apply(coefficients_concatJ_5_00h[, 1:26], 2, var)


print(variances_J_5_00h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_5_00h.csv"


write.csv2(variances_J_5_00h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_5_00h)


plot(1:length(variances_J_5_00h), variances_J_5_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_5_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels













##12H


### Nombre de profils dans chaque groupe





na_counts0 = apply(jour512, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}


##Rcherche des profils de longueur maximale

Base19 = jour512[apply(jour512, 1, function(row) sum(is.na(row)) == 1), ]


Base19= Base19[, -ncol(Base19)]

nbasis_values = seq(5, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base19))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")








# Boucle pour créer Base1 à Base29
bases = list()


for (i in 1:29) {
  
  base_temp = jour512[apply(jour512, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_5_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_5_12h)) {
    coefficients_concatJ_5_12h = coefficients_temp  
  } else {
    coefficients_concatJ_5_12h = rbind(coefficients_concatJ_5_12h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_5_12h.csv"


write.csv2(coefficients_concatJ_5_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_5_12h)

head(coefficients_concatJ_5_12h)

### Variances des coeffs de la matrice concatenée

variances_J_5_12h = apply(coefficients_concatJ_5_12h[, 1:26], 2, var)


print(variances_J_5_12h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_4_12h.csv"


write.csv2(variances_J_3_12h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_3_12h)


plot(1:length(variances_J_3_12h), variances_J_3_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_3_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels






###SIXIEME JOUR

##00H

### Nombre de profils dans chaque groupe





na_counts0 = apply(jour600, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}

##Rcherche des profils de longueur maximale

Base21 = jour600[apply(jour600, 1, function(row) sum(is.na(row)) == 1), ]


Base21 = Base21[, -ncol(Base21)]

nbasis_values = seq(5, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base21))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")


 





# Boucle pour créer Base1 à Base29
bases = list()


for (i in 1:29) {
  
  base_temp = jour600[apply(jour600, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_6_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_6_00h)) {
    coefficients_concatJ_6_00h = coefficients_temp  
  } else {
    coefficients_concatJ_6_00h = rbind(coefficients_concatJ_6_00h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_6_00h.csv"


write.csv2(coefficients_concatJ_6_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_6_00h)


dim(coefficients_concatJ_6_00h)

head(coefficients_concatJ_6_00h)

### Variances des coeffs de la matrice concatenée

variances_J_6_00h = apply(coefficients_concatJ_6_00h[, 1:26], 2, var)


print(variances_J_6_00h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_6_00h.csv"


write.csv2(variances_J_6_00h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_6_00h)


plot(1:length(variances_J_6_00h), variances_J_6_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_6_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels













##12H


### Nombre de profils dans chaque groupe





na_counts0 = apply(jour612, 1, function(x) sum(is.na(x)))


na_summary0= table(na_counts0)


na_summary_df0= as.data.frame(na_summary0)

na_summary_df0







###validation
compute_cv_error_bspline = function(nbasis, argvals, y, K = 10) {
  n = ncol(y)  
  fold_size = ceiling(n / K)
  errors = c()
  
  for (k in 1:K) {
    test_indices = ((k - 1) * fold_size + 1):min(k * fold_size, n)
    train_indices = setdiff(1:n, test_indices)
    
    y_train = y[, train_indices, drop = FALSE]
    y_test = y[, test_indices, drop = FALSE]
    
    basis = create.bspline.basis(rangeval = range(argvals), nbasis = nbasis)
    fd_train = Data2fd(argvals = argvals, y = y_train, basisobj = basis)
    
    
    y_pred = eval.fd(argvals, fd_train)
    
    
    valid_indices = test_indices[test_indices <= ncol(y_pred)]
    y_test_pred = y_pred[, valid_indices, drop = FALSE]
    
    
    fold_error = sum((y[, valid_indices, drop = FALSE] - y_test_pred)^2)
    errors = c(errors, fold_error)
    
    
    cat("Fold:", k, "Test Indices:", test_indices, "Fold Error:", fold_error, "\n")
  }
  
  return(mean(errors))
}

##Rcherche des profils de longueur maximale

Base23 = jour512[apply(jour512, 1, function(row) sum(is.na(row)) == 1), ]


Base23 = Base23[, -ncol(Base23)]

nbasis_values = seq(5, 30, by = 1)


cv_errors = sapply(nbasis_values, function(n) {
  cat("Calculating CV error for nbasis =", n, "\n")
  compute_cv_error_bspline(n, densite, t(Base23))
})


optimal_nbasis = nbasis_values[which.min(cv_errors)]
print(paste("Nombre optimal de fonctions de base:", optimal_nbasis))##on a 26


optimal_basis1 = create.bspline.basis(rangeval = c(0, 31), nbasis = optimal_nbasis)


vitesse_fd_optimal1 = Data2fd(densite, t(Base0), optimal_basis1)





plot(vitesse_fd_optimal1, xlab = "Densite", ylab = "Vitesse")


 





# Boucle pour créer Base1 à Base29
bases = list()


for (i in 1:29) {
  
  base_temp = jour612[apply(jour612, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 26)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_6_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_6_12h)) {
    coefficients_concatJ_6_12h = coefficients_temp  
  } else {
    coefficients_concatJ_6_12h = rbind(coefficients_concatJ_6_12h, coefficients_temp)  
  }
}

### fichier csv de la matrice concatenée


file_path = "C:/Users/djafa/Desktop/Nouveaustage/coefficients_concatJ_6_12h.csv"


write.csv2(coefficients_concatJ_6_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_6_12h)

head(coefficients_concatJ_6_12h)

### Variances des coeffs de la matrice concatenée

variances_J_6_12h = apply(coefficients_concatJ_6_12h[, 1:26], 2, var)


print(variances_J_6_12h)


file_path = "C:/Users/djafa/Desktop/Nouveaustage/variances_J_6_12h.csv"


write.csv2(variances_J_6_12h, file = file_path, row.names = TRUE)


### Représentation des varianves

bspl4_names = names(variances_J_6_12h)


plot(1:length(variances_J_6_12h), variances_J_6_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_6_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels





################################### les Boxplots des coeffs pour les 12 fichiers#####################


##Jour1
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_1_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_1_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:26))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(26)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  







##Jour2
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_2_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_2_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:26))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(26)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  













##Jour3
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_3_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_3_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  


##Jour4
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_4_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_4_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  






##Jour5
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_5_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_5_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  








##Jour6
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_6_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_6_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  




############### Matrices de corrélation des coeffs pour les 12 fichiers############



###Premier jour
#00h
correlation_matrix_1_00 =cor(coefficients_concatJ_1_00h)

par(mfrow=c(1, 1))


corrplot(correlation_matrix_1_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_1_12 =cor(coefficients_concatJ_1_12h)

corrplot(correlation_matrix_1_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


###Deuxième jour
#00h
correlation_matrix_2_00 =cor(coefficients_concatJ_2_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_2_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_2_12 =cor(coefficients_concatJ_2_12h)

corrplot(correlation_matrix_2_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)



###Trosième jour
#00h
correlation_matrix_3_00 =cor(coefficients_concatJ_3_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_3_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_3_12 =cor(coefficients_concatJ_3_12h)

corrplot(correlation_matrix_3_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)



###Quatrième jour
#00h
correlation_matrix_4_00 =cor(coefficients_concatJ_4_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_4_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_4_12 =cor(coefficients_concatJ_4_12h)

corrplot(correlation_matrix_4_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)










###Cinquième jour
#00h
correlation_matrix_5_00 =cor(coefficients_concatJ_5_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_5_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_5_12 =cor(coefficients_concatJ_5_12h)

corrplot(correlation_matrix_5_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)



###Sixième jour
#00h
correlation_matrix_6_00 =cor(coefficients_concatJ_6_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_6_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_6_12 =cor(coefficients_concatJ_6_12h)

corrplot(correlation_matrix_6_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)




##################################### Ici nous avons fixé le nombre optimal de transformation à 27

##Premier jour 
#12h

#Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour112[apply(jour112, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_1_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_1_12h)) {
    coefficients_concatJ_1_12h = coefficients_temp  
  } else {
    coefficients_concatJ_1_12h = rbind(coefficients_concatJ_1_12h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_1_12h.csv"


write.csv2(coefficients_concatJ_1_12h, file = file_path, row.names = TRUE)#,sep = ";"

#dim(coefficients_concatJ_1_12h)


#dim(coefficients_concatJ_1_12h)





variances_J_1_12h = apply(coefficients_concatJ_1_12h[, 1:27], 2, var)


print(variances_J_1_12h)

file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_1_12h.csv"


write.csv2(variances_J_1_12h, file = file_path, row.names = TRUE)




#00h







# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour100[apply(jour100, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_1_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_1_00h)) {
    coefficients_concatJ_1_00h = coefficients_temp  
  } else {
    coefficients_concatJ_1_00h = rbind(coefficients_concatJ_1_00h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_1_00h.csv"


write.csv2(coefficients_concatJ_1_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_1_00h)


dim(coefficients_concatJ_1_00h)





variances_J_1_00h = apply(coefficients_concatJ_1_00h[, 1:27], 2, var)


print(variances_J_1_00h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_1_00h.csv"


write.csv2(variances_J_1_00h, file = file_path, row.names = TRUE)







##Deuxième jour





#00h



# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour100[apply(jour100, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_2_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_2_00h)) {
    coefficients_concatJ_2_00h = coefficients_temp  
  } else {
    coefficients_concatJ_2_00h = rbind(coefficients_concatJ_2_00h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_2_00h.csv"


write.csv2(coefficients_concatJ_2_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_2_00h)


dim(coefficients_concatJ_2_00h)





variances_J_2_00h = apply(coefficients_concatJ_2_00h[, 1:27], 2, var)


print(variances_J_2_00h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_2_00h.csv"


write.csv2(variances_J_2_00h, file = file_path, row.names = TRUE)






#12h



# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour212[apply(jour212, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_2_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_2_12h)) {
    coefficients_concatJ_2_12h = coefficients_temp  
  } else {
    coefficients_concatJ_2_12h = rbind(coefficients_concatJ_2_12h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_2_12h.csv"


write.csv2(coefficients_concatJ_2_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_2_12h)




head(coefficients_concatJ_2_12h)



variances_J_2_12h = apply(coefficients_concatJ_2_12h[, 1:27], 2, var)


print(variances_J_2_12h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_2_12h.csv"


write.csv2(variances_J_2_12h, file = file_path, row.names = TRUE)





##Troisième jour

#00h










# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour300[apply(jour300, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_3_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_3_00h)) {
    coefficients_concatJ_3_00h = coefficients_temp  
  } else {
    coefficients_concatJ_3_00h = rbind(coefficients_concatJ_3_00h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_3_00h.csv"


write.csv2(coefficients_concatJ_3_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_3_00h)


dim(coefficients_concatJ_3_00h)

head(coefficients_concatJ_3_00h)



variances_J_3_00h = apply(coefficients_concatJ_3_00h[, 1:27], 2, var)


print(variances_J_3_00h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_3_00h.csv"


write.csv2(variances_J_1_00h, file = file_path, row.names = TRUE)





#12h


# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour312[apply(jour312, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_3_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_3_12h)) {
    coefficients_concatJ_3_12h = coefficients_temp  
  } else {
    coefficients_concatJ_3_12h = rbind(coefficients_concatJ_3_12h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_3_12h.csv"


write.csv2(coefficients_concatJ_3_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_3_12h)

head(coefficients_concatJ_3_12h)



variances_J_3_12h = apply(coefficients_concatJ_3_12h[, 1:27], 2, var)


print(variances_J_3_12h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_3_12h.csv"


write.csv2(variances_J_3_12h, file = file_path, row.names = TRUE)






#Quatrième jour
#00h










# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour400[apply(jour400, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_4_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_4_00h)) {
    coefficients_concatJ_4_00h = coefficients_temp  
  } else {
    coefficients_concatJ_4_00h = rbind(coefficients_concatJ_4_00h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_4_00h.csv"


write.csv2(coefficients_concatJ_4_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_4_00h)


dim(coefficients_concatJ_4_00h)

head(coefficients_concatJ_4_00h)



variances_J_4_00h = apply(coefficients_concatJ_4_00h[, 1:27], 2, var)


print(variances_J_4_00h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_4_00h.csv"


write.csv2(variances_J_4_00h, file = file_path, row.names = TRUE)





#12h






# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:28) {
  
  base_temp = jour412[apply(jour412, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_4_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_4_12h)) {
    coefficients_concatJ_4_12h = coefficients_temp  
  } else {
    coefficients_concatJ_4_12h = rbind(coefficients_concatJ_4_12h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_4_12h.csv"


write.csv2(coefficients_concatJ_4_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_4_12h)

head(coefficients_concatJ_4_12h)



variances_J_4_12h = apply(coefficients_concatJ_4_12h[, 1:27], 2, var)


print(variances_J_4_12h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_4_12h.csv"


write.csv2(variances_J_4_12h, file = file_path, row.names = TRUE)



##cinquième jour
#00h














# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:29) {
  
  base_temp = jour500[apply(jour500, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_5_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_5_00h)) {
    coefficients_concatJ_5_00h = coefficients_temp  
  } else {
    coefficients_concatJ_5_00h = rbind(coefficients_concatJ_5_00h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_5_00h.csv"


write.csv2(coefficients_concatJ_5_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_5_00h)


dim(coefficients_concatJ_5_00h)

head(coefficients_concatJ_5_00h)



variances_J_5_00h = apply(coefficients_concatJ_5_00h[, 1:27], 2, var)


print(variances_J_5_00h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_5_00h.csv"


write.csv2(variances_J_5_00h, file = file_path, row.names = TRUE)




#12h














# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:29) {
  
  base_temp = jour512[apply(jour512, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_5_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_5_12h)) {
    coefficients_concatJ_5_12h = coefficients_temp  
  } else {
    coefficients_concatJ_5_12h = rbind(coefficients_concatJ_5_12h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_5_12h.csv"


write.csv2(coefficients_concatJ_5_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_5_12h)

head(coefficients_concatJ_5_12h)



variances_J_5_12h = apply(coefficients_concatJ_5_12h[, 1:27], 2, var)


print(variances_J_5_12h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_5_12h.csv"


write.csv2(variances_J_5_12h, file = file_path, row.names = TRUE)


##6 eme jour
#00h




# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:29) {
  
  base_temp = jour600[apply(jour600, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_6_00h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_6_00h)) {
    coefficients_concatJ_6_00h = coefficients_temp  
  } else {
    coefficients_concatJ_6_00h = rbind(coefficients_concatJ_6_00h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_6_00h.csv"


write.csv2(coefficients_concatJ_6_00h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_6_00h)


dim(coefficients_concatJ_6_00h)

head(coefficients_concatJ_6_00h)



variances_J_6_00h = apply(coefficients_concatJ_6_00h[, 1:27], 2, var)


print(variances_J_6_00h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_6_00h.csv"


write.csv2(variances_J_6_00h, file = file_path, row.names = TRUE)





#12h

















# Initialiser une liste pour stocker les bases de données
bases = list()

# Boucle pour créer Base1 à Base28
for (i in 1:29) {
  
  base_temp = jour612[apply(jour612, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:29) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}


##Concaténation
coefficients_concatJ_6_12h = NULL


for (i in 1:length(vitesse_fd_list)) {
  
  coefficients_temp = coef(vitesse_fd_list[[i]])
  coefficients_temp = t(coefficients_temp)  
  
  
  if (is.null(coefficients_concatJ_6_12h)) {
    coefficients_concatJ_6_12h = coefficients_temp  
  } else {
    coefficients_concatJ_6_12h = rbind(coefficients_concatJ_6_12h, coefficients_temp)  
  }
}

###csv


file_path = "C:/Users/djafa/Desktop/FICHIER/coefficients_concatJ_6_12h.csv"


write.csv2(coefficients_concatJ_6_12h, file = file_path, row.names = TRUE)

dim(coefficients_concatJ_6_12h)

head(coefficients_concatJ_6_12h)



variances_J_6_12h = apply(coefficients_concatJ_6_12h[, 1:27], 2, var)


print(variances_J_6_12h)


file_path = "C:/Users/djafa/Desktop/FICHIER/variances_J_6_12h.csv"


write.csv2(variances_J_6_12h, file = file_path, row.names = TRUE)



head(coefficients_concatJ_1_12h)






#########################Boxplot sur les transformations avec 27 éléments fixés#############


##Jour1
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_1_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_1_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  










##Jour2
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_2_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_2_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  













##Jour3
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_3_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_3_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  































##Jour4
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_4_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_4_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  






##Jour5
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_5_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_5_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  























##Jour6
##00H

par(mfrow=c(1, 2))




dat = as_tibble(coefficients_concatJ_6_00h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

##12h

dat = as_tibble(coefficients_concatJ_6_12h)


data_long = pivot_longer(dat, cols = everything(), names_to = "Variable", values_to = "Valeur")


data_long$Variable = factor(data_long$Variable, levels = paste0("bspl4.", 1:27))


ggplot(data_long, aes(x = Variable, y = Valeur, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       v                                   
scale_fill_manual(values = rainbow(27)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  


############################# Matrices de corrélation###########################
###Premier jour
#00h
correlation_matrix_1_00 =cor(coefficients_concatJ_1_00h)

par(mfrow=c(1, 1))


corrplot(correlation_matrix_1_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_1_12 =cor(coefficients_concatJ_1_12h)

corrplot(correlation_matrix_1_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


###Deuxième jour
#00h
correlation_matrix_2_00 =cor(coefficients_concatJ_2_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_2_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_2_12 =cor(coefficients_concatJ_2_12h)

corrplot(correlation_matrix_2_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)



###Trosième jour
#00h
correlation_matrix_3_00 =cor(coefficients_concatJ_3_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_3_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_3_12 =cor(coefficients_concatJ_3_12h)

corrplot(correlation_matrix_3_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)



###Quatrième jour
#00h
correlation_matrix_4_00 =cor(coefficients_concatJ_4_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_4_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_4_12 =cor(coefficients_concatJ_4_12h)

corrplot(correlation_matrix_4_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)










###Cinquième jour
#00h
correlation_matrix_5_00 =cor(coefficients_concatJ_5_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_5_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h
correlation_matrix_5_12 =cor(coefficients_concatJ_5_12h)

corrplot(correlation_matrix_5_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)



###Sixième jour
#00h
correlation_matrix_6_00 =cor(coefficients_concatJ_6_00h)


par(mfrow=c(1, 1))


corrplot(correlation_matrix_6_00, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)


#12h


correlation_matrix_6_12 =cor(coefficients_concatJ_6_12h)

corrplot(correlation_matrix_6_12, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 27)




###############  Variances ###########################################
##1er

#00h


variances_J_1_00h = apply(coefficients_concatJ_1_00h[, 1:27], 2, var)


print(variances_J_1_00h)



bspl4_names = names(variances_J_1_00h)


plot(1:length(variances_J_1_00h), variances_J_1_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_1_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



#12h


variances_J_1_12h = apply(coefficients_concatJ_1_12h[, 1:27], 2, var)


print(variances_J_1_12h)



bspl4_names = names(variances_J_1_12h)


plot(1:length(variances_J_1_12h), variances_J_1_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_1_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



















##2eme

#00h


variances_J_2_00h = apply(coefficients_concatJ_2_00h[, 1:27], 2, var)


print(variances_J_2_00h)



bspl4_names = names(variances_J_2_00h)


plot(1:length(variances_J_2_00h), variances_J_2_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_2_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



#12h


variances_J_1_12h = apply(coefficients_concatJ_2_12h[, 1:27], 2, var)


print(variances_J_2_12h)



bspl4_names = names(variances_J_2_12h)


plot(1:length(variances_J_2_12h), variances_J_2_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_2_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels




























##3eme

#00h


variances_J_3_00h = apply(coefficients_concatJ_3_00h[, 1:27], 2, var)


print(variances_J_3_00h)



bspl4_names = names(variances_J_3_00h)


plot(1:length(variances_J_3_00h), variances_J_3_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_3_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



#12h


variances_J_3_12h = apply(coefficients_concatJ_3_12h[, 1:27], 2, var)


print(variances_J_3_12h)



bspl4_names = names(variances_J_3_12h)


plot(1:length(variances_J_3_12h), variances_J_3_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_3_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels


















##4eme

#00h


variances_J_4_00h = apply(coefficients_concatJ_4_00h[, 1:27], 2, var)


print(variances_J_4_00h)



bspl4_names = names(variances_J_4_00h)


plot(1:length(variances_J_4_00h), variances_J_4_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_4_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



#12h


variances_J_4_12h = apply(coefficients_concatJ_4_12h[, 1:27], 2, var)


print(variances_J_4_12h)



bspl4_names = names(variances_J_4_12h)


plot(1:length(variances_J_4_12h), variances_J_4_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_4_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels














##5eme

#00h


variances_J_5_00h = apply(coefficients_concatJ_5_00h[, 1:27], 2, var)


print(variances_J_5_00h)



bspl4_names = names(variances_J_5_00h)


plot(1:length(variances_J_5_00h), variances_J_5_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_5_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



#12h


variances_J_5_12h = apply(coefficients_concatJ_5_12h[, 1:27], 2, var)


print(variances_J_5_12h)



bspl4_names = names(variances_J_5_12h)


plot(1:length(variances_J_5_12h), variances_J_5_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_5_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels






##5eme

#00h


variances_J_6_00h = apply(coefficients_concatJ_6_00h[, 1:27], 2, var)


print(variances_J_6_00h)



bspl4_names = names(variances_J_6_00h)


plot(1:length(variances_J_6_00h), variances_J_6_00h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_6_00h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



#12h


variances_J_6_12h = apply(coefficients_concatJ_6_12h[, 1:27], 2, var)


print(variances_J_6_12h)



bspl4_names = names(variances_J_6_12h)


plot(1:length(variances_J_6_12h), variances_J_6_12h, type = "b", xlab = "bspl4.i", ylab = "Variance", pch = 10, col = "blue", xaxt = "n")


axis(1, at = 1:length(variances_J_6_12h), labels = bspl4_names, las = 2, cex.axis = 1) # las=2 pour faire pivoter les labels



##########Notre dernier travail s'est concentré sur tous les profils du premier jour à 12h transformer avec 26 éléments##########



# Boucle pour créer Base1 à Base28
bases = list()


for (i in 1:28) {
  
  base_temp = jour112[apply(jour112, 1, function(row) sum(is.na(row)) == i), ]
  
  
  for (j in 1:i) {
    base_temp = base_temp[, -ncol(base_temp)]
  }
  
  
  bases[[paste0("Base", i)]] = base_temp
}

library(fda)


vitesse_fd_list = list()


for (i in 1:28) {
  
  
  base_temp = bases[[paste0("Base", i)]]
  
  
  optimal_basis_temp = create.bspline.basis(rangeval = c(0, 32 - i), nbasis = 27)
  
  
  densite_temp = 0:(32 - i)
  
  
  vitesse_fd_list[[paste0("vitesse_fd_optimal", i)]] = Data2fd(densite_temp, t(base_temp), optimal_basis_temp)
  
}

# Boucle pour tracer toutes les données fonctionnels 
par(mfrow = c(1, 1))
                

plot(vitesse_fd_list[[1]], xlab = "Densité", ylab = "Vitesse",  main = "Représentation des Profils Fonctionnels (Bases 1 à 28)")

for (i in 2:28) {lines(vitesse_fd_list[[i]])}





# Boucle pour tracer les données fonctionnelles par groupe de meme longueur
for (i in seq(1, 28, by = 4)) {
  
  
  par(mfrow = c(2, 2))
  
  
  for (j in 0:3) {
    if (i + j <= 28) { 
      plot(vitesse_fd_list[[i + j]], 
           xlab = "Densité", ylab = "Vitesse", 
           main = paste("longueur", 33 - (i + j)))  
    }
  }
  
  
  readline(prompt = "Appuyez sur [Entrée] pour afficher le groupe suivant...")
}













