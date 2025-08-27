import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Nümunə Datasetlər
# ------------------------------
clothes = pd.DataFrame({
    "itemId": [1, 2, 3, 4, 5],
    "title": ["T-shirt", "Jeans", "Sweater", "Jacket", "Sneakers"]
})

ratings = pd.DataFrame({
    "userId": [1, 1, 2, 2, 3, 3, 4, 6, 6],
    "itemId": [1, 2, 2, 3, 1, 4, 5, 3, 5],
    "rating": [5, 4, 5, 3, 4, 5, 5, 5, 4],
    "timestamp": [111, 112, 113, 114, 115, 116, 117, 118, 119]
})

# ------------------------------
# Item-Based Recommendation
# ------------------------------
user = 6  # Tövsiyə ediləcək istifadəçi

# Məhsullarla reytinqləri birləşdir
df = clothes.merge(ratings, how="left", on="itemId")

# İstifadəçinin ən son 5 verdiyi məhsulun id-sini tapmaq
cloth_id = ratings[(ratings["userId"] == user) & (ratings["rating"] == 5.0)] \
    .sort_values(by="timestamp", ascending=False)["itemId"].values[0]

last_cloth = clothes[clothes["itemId"] == cloth_id]["title"].values[0]
print("İstifadəçinin ən son bəyəndiyi məhsul:", last_cloth)

# User-Item Matrix
user_cloth_df = df.pivot_table(index="userId", columns="title", values="rating")

# NaN-ları 0 ilə doldur
user_cloth_df_filled = user_cloth_df.fillna(0)

# Məhsullar arası oxşarlıq (Cosine Similarity)
item_similarity = cosine_similarity(user_cloth_df_filled.T)

# DataFrame formatına salaq
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_cloth_df_filled.columns,
    columns=user_cloth_df_filled.columns
)

# Son seçilən məhsula görə oxşar məhsulları sırala
similarity_series = item_similarity_df[last_cloth].sort_values(ascending=False)

print("\nƏn oxşar məhsullar:\n", similarity_series.head(10))
