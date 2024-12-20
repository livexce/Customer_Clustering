import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk

# Télécharger les ressources nécessaires pour l'analyse de sentiment
nltk.download('vader_lexicon')

# Charger les données
data = pd.read_csv("E-commerce.csv")

# Nettoyer les données
data = data.dropna()
print(f"Données nettoyées : {data.shape[0]} lignes restantes.")

# **1. Analyse de Sentiments**
analyzer = SentimentIntensityAnalyzer()

# Appliquer l'analyse de sentiments à chaque avis
def analyze_sentiment(review):
    if isinstance(review, str):
        return analyzer.polarity_scores(review)['compound']  # Score global
    return 0

data['Sentiment Score'] = data['Product Reviews'].apply(analyze_sentiment)
print(data[['Product Reviews', 'Sentiment Score']].head())

# **2. Extraction de Thèmes avec TF-IDF**
# Vectoriser les avis pour extraire les mots-clés
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = tfidf.fit_transform(data['Product Reviews'].fillna(""))

# Ajouter les mots-clés comme colonnes
keywords = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=keywords)
data = pd.concat([data, tfidf_df], axis=1)
print("Mots-clés extraits : ", keywords)

# **3. Clustering**
# Sélectionner les colonnes pour le clustering
selected_data = data[['Annual Income', 'Time on Site', 'Age', 'Sentiment Score'] + list(keywords)]

# Normalisation
scaler = StandardScaler()
data_scaled = scaler.fit_transform(selected_data)

# Méthode du coude
distortions = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    distortions.append(kmeans.inertia_)

plt.plot(range(2, 10), distortions, marker='o')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Distorsion')
plt.title('Méthode du Coude (Avec Sentiment)')
plt.show()

# Appliquer K-Means avec 4 clusters (ou le nombre choisi via la méthode du coude)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data['Cluster'] = clusters

# **4. Visualisation des Clusters**
plt.scatter(data['Annual Income'], data['Time on Site'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Time on Site')
plt.title('Clusters K-Means (Enrichis)')
plt.show()

# Vérifier les clusters
print(data[['Annual Income', 'Time on Site', 'Age', 'Sentiment Score', 'Cluster']].head())
