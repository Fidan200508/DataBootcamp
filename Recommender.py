import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Dataset (nümunə üçün)
data = {
    "id": [1, 2, 3, 4, 5],
    "icerik": [
        "Python proqramlaşdırma dili çox güclüdür",
        "Java proqramlaşdırma dili də geniş istifadə olunur",
        "Python maşın öyrənməsində məşhurdur",
        "C++ oyun inkişafında geniş istifadə olunur",
        "JavaScript veb proqramlaşdırmada əsas dildir"
    ]
}

df = pd.DataFrame(data)

########################
# 2. TF-IDF
########################
tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['icerik'])

########################
# 3. Similarity
########################
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

########################
# 4. Benzer Sorguların Getirilməsi
########################
def recommender(product_idx, top_n=5):
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # özünü çıxmaq üçün 1-dən başlayırıq
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# İstifadə nümunəsi
print("Sorğu: ", df.iloc[0]["icerik"])
print("\nOxşar nəticələr:\n", recommender(0, 3))
