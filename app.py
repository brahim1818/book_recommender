from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Chargement des données
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
user_similarity_scores = pickle.load(open('user_similarity_scores.pkl', 'rb'))  # Charger les scores de similarité utilisateur

# Convertir la matrice pivot en matrice creuse
sparse_matrix = csr_matrix(pt.values)
# Appliquer KNN
knn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn.fit(sparse_matrix)

# Fonction principale qui calcule les détails de quatre livres similaires
def recommendd(book_name):
    # Récupérer l'index du livre
    if book_name not in pt.index:
        return f"Le livre '{book_name}' n'existe pas dans le système."
    
    index = np.where(pt.index == book_name)[0][0]
    
    # Calculer les similarités
    distances, indices = knn.kneighbors(sparse_matrix[index], n_neighbors=11)  # Utiliser plus de voisins

    data = []
    for i in range(len(indices.flatten())):
        item = []
        temp_df = books[books['Book-Title'] == pt.index[indices.flatten()[i]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor

# 1. Créer des embeddings (en utilisant SVD)
svd = TruncatedSVD(n_components=50)  # Choisissez le nombre de dimensions
user_embeddings = svd.fit_transform(pt.T)  # Transposez pt pour que les utilisateurs soient en lignes
book_embeddings = svd.components_.T

# 2. Préparer les données d'entraînement
data = []
for user_idx in range(len(user_embeddings)):
    for book_idx in range(len(book_embeddings)):
        rating = pt.iloc[book_idx, user_idx]  # Ajustez l'indexation pour la matrice transposée
        if rating > 0:  # Vérifiez que la note est positive
            data.append((user_embeddings[user_idx], book_embeddings[book_idx], rating))

# 3. Entraînement du modèle ANN
X_train, y_train = zip(*[(np.concatenate((user, book)), rating) for user, book, rating in data])
X_train = np.array(X_train)
y_train = np.array(y_train)

modelmlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
modelmlp.fit(X_train, y_train)

# Récupérer les IDs de livres uniques
book_ids = np.unique(books['Book-Title'])

def recommend_with_embeddings(user_id, n_recommendations=5):
    user_id = int(user_id)
    user_embedding = user_embeddings[user_id]
    predictions = []

    # Prédire les évaluations pour tous les livres
    for book_idx in range(len(book_embeddings)):
        book_embedding = book_embeddings[book_idx]
        predicted_rating = modelmlp.predict([np.concatenate((user_embedding, book_embedding))])
        predictions.append((book_idx, predicted_rating[0]))  # Utiliser la première valeur de la prédiction

    # Obtenir les indices des livres les mieux notés
    recommended_indices = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Récupérer les informations sur les livres recommandés
    data = []
    for idx, rating in recommended_indices:
        item = []

        # Récupérer le titre du livre à partir de book_ids
        book_title = book_ids[idx]  # Utiliser l'index pour obtenir le titre
        print(book_title)

        # Extraire les informations du DataFrame des livres
        temp_df = books[books['Book-Title'] == book_title]

        if not temp_df.empty:  # Vérifier que le DataFrame n'est pas vide
            item.extend(list(temp_df['Book-Title'].values))
            item.extend(list(temp_df['Book-Author'].values))
            item.extend(list(temp_df['Image-URL-M'].values))
        
            data.append(item)

    return data





app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    
    recommendation_type = request.form.get('recommendation_type')  # Obtenir le type de recommandation

    if recommendation_type == 'item-item':
        print(f"User Input: {user_input}") 
        index = np.where(pt.index == user_input)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)

        return render_template('recommend.html', data=data)

    elif recommendation_type == 'user-user':
        user_id = request.form.get('user_input')  # Obtenir l'ID utilisateur directement depuis le formulaire
        user_id_numeric = int(user_id)
        index = np.where(pt.columns == user_id_numeric)[0][0]
        print(f"User ID entered: {index}")
        indice = int(index)
        similar_users = sorted(list(enumerate(user_similarity_scores[indice])), key=lambda x: x[1], reverse=True)[1:6]
        print(f"su: {similar_users}")
        data = []
        for i in similar_users:
            
            similar_user_id = pt.columns[i[0]]
            rated_books = pt[similar_user_id][pt[similar_user_id] > 5].index.tolist()[1:2]
            
            for book in rated_books:
                temp_df = books[books['Book-Title'] == book]
                item = []
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
                data.append(item)

        # Supprimer les doublons
        unique_data = [list(x) for x in set(tuple(x) for x in data)]
        
        
        return render_template('recommend.html', data=unique_data)  # Retourner les données à afficher
    elif recommendation_type == 'knn':
        data = recommendd(user_input)  # Appeler la nouvelle fonction KNN
        return render_template('recommend.html', data=data)
    elif recommendation_type == 'ann':
        data = recommend_with_embeddings(user_input)  # Appeler la nouvelle fonction KNN
        return render_template('recommend.html', data=data)
    elif recommendation_type == 'annt':
        data = recommend_with_embeddings1(user_input)  # Appeler la nouvelle fonction KNN
        return render_template('recommend.html', data=data)
        
if __name__ == '__main__':
    app.run(debug=True)