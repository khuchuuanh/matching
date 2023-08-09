import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


df1 = pd.read_excel("D:\matching\Jente dataa.xlsx") 
df2 = pd.read_excel("D:\matching\Matched_Jente Data (1).xlsx", skiprows=4)

df1['competitor'] = df1["competitor"].str.lower().str.replace(r"\s|\.|'|&|/|-", "", regex=True)
df2['goods_nm'] = df2['goods_nm'].apply(lambda x: re.sub(r"\s*\/\s*.+", "", x))
categories1 = df1["competitor"].dropna(axis= 0)

categories1 = df1["competitor"].unique()
categories2 = df2["brand"].unique()
matching_categories = set(categories1).intersection(categories2)
merged_rows = []

for category in matching_categories:
    df1_category = df1[df1["competitor"] == category]
    df2_category = df2[df2["brand"] == category]

    goods1 = df1_category["good_nm"].dropna().unique()
    goods2 = df2_category["goods_nm"].dropna().unique()
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(goods1)
    similarities = cosine_similarity(tfidf, vectorizer.transform(goods2))
    
    matching_pairs = zip(*similarities.nonzero())
    for i, u in matching_pairs:
        if similarities[i, u] > 0.8:
            new_row = pd.concat([df1_category[df1_category["good_nm"] == goods1[i]].reset_index(drop=True),
                                 df2_category[df2_category["goods_nm"] == goods2[u]].reset_index(drop=True)], axis=1)
            merged_rows.append(new_row)

merged_df = pd.concat(merged_rows, ignore_index=True)

merged_df = merged_df.dropna(how="all", axis=0)
merged_df = merged_df[["ID", "competitor", "good_opt", "good_nm", "sale_price", "option_nm", "goods_nm", "brand", "goods_opt"]]
merged_df.to_excel('merged_data_big.xlsx', index=False)

print(merged_df)





# for category in matching_categories:
#     df1_category = df1[df1["competitor"] == category]
#     df2_category = df2[df2["brand"] == category]
    
#     goods1 = df1_category["good_nm"].unique()
#     goods2 = df2_category["goods_nm"].unique()
#     for i in goods1:
#             for u in goods2:
#                 vectorizer = TfidfVectorizer()
#                 tfidf = vectorizer.fit_transform([i, u])
#                 similarity= cosine_similarity(tfidf[0], tfidf[1])
#                 if similarity > 0.6:
#                     new_row = pd.concat([df1_category[df1_category["good_nm"] == i].reset_index(drop=True),
#                                      df2_category[df2_category["goods_nm"] == u].reset_index(drop=True)],axis=1)
#                     merged_df = merged_df.append(new_row, ignore_index=True)


# merged_df = merged_df.dropna(how ="all" ,axis= 0)
# merged_df = merged_df[["ID", "competitor", "good_opt", "good_nm", "sale_price", "option_nm","goods_nm","brand","goods_opt"]]
# merged_df.to_excel('merged_data.xlsx', index=False)

# print(merged_df)
                     