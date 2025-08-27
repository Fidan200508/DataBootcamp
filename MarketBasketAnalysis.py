import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# 1. Nümunə dataset yaradırıq
# -----------------------------
data = {
    "Transaction": [1,1,2,2,3,3,3,4,4,5,5],
    "date_time": ["2024-08-01","2024-08-01","2024-08-02","2024-08-02",
                  "2024-08-03","2024-08-03","2024-08-03",
                  "2024-08-04","2024-08-04",
                  "2024-08-05","2024-08-05"],
    "Item": ["Milk","Bread","Milk","Eggs",
             "Bread","Butter","Milk",
             "Beer","Chips",
             "Milk","Chips"]
}

df = pd.DataFrame(data)

# Convert data types
df["date_time"] = pd.to_datetime(df["date_time"])
df["Item"] = df["Item"].astype(str)

print("Input Data")
print(df)

# -----------------------------
# 2. Transaction–Item matrix
# -----------------------------
basket = (df.groupby(["Transaction","Item"])["Item"]
            .count()
            .unstack()
            .fillna(0)
            .applymap(lambda x: 1 if x > 0 else 0))

print("\nTransaction–Item Matrix")
print(basket)

# -----------------------------
# 3. Frequent itemsets (Apriori)
# -----------------------------
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

print("\nFrequent Itemsets")
print(frequent_itemsets)

# -----------------------------
# 4. Association Rules
# -----------------------------
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("\nAssociation Rules")
print(rules[["antecedents","consequents","support","confidence","lift"]])

# -----------------------------
# 5. Recommender function
# -----------------------------
def arl_recommender(rules_df, product, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendations = []
    for _, row in sorted_rules.iterrows():
        if product in row["antecedents"]:
            recommendations.extend(list(row["consequents"]))
    return list(set(recommendations))[:rec_count]

print("\nRecommendation for 'Milk'")
print(arl_recommender(rules, "Milk", 2))
