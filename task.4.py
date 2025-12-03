# =============================
# Task-04: Social Media Sentiment Analysis
# =============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
df = pd.read_csv("combined_chennai.csv.zip")  # apni file ka path
print("✅ Dataset loaded successfully")
print(df.head())

# ----------------------------
# 2️⃣ Check null values
# ----------------------------
print("\nNull values in each column before filling:")
print(df.isnull().sum())

# ----------------------------
# 3️⃣ Fill null values
# ----------------------------
# String columns ke liye NaN ko empty string se replace karo
string_cols = ['outlinks', 'tcooutlinks', 'media', 'retweetedTweet', 
               'quotedTweet', 'inReplyToTweetId', 'inReplyToUser', 
               'mentionedUsers', 'hashtags', 'cashtags', 'content']

for col in string_cols:
    if col in df.columns:
        df[col] = df[col].fillna('')

# Numeric columns ke liye NaN ko 0 se replace karo
numeric_cols = ['replyCount', 'retweetCount', 'likeCount', 'quoteCount']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print("\nNull values in each column after filling:")
print(df.isnull().sum())

# ----------------------------
# 4️⃣ Sentiment Analysis
# ----------------------------
df['polarity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

def sentiment_label(p):
    if p > 0:
        return 'Positive'
    elif p < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(sentiment_label)

print("\n✅ Sentiment analysis done")
print(df[['content','polarity','subjectivity','sentiment']].head())

# ----------------------------
# 5️⃣ Sentiment Distribution (Bar chart)
# ----------------------------
plt.style.use('dark_background')
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='sentiment', palette='Blues')
plt.title("Sentiment Distribution", color='cyan')
plt.show()

# ----------------------------
# 6️⃣ Polarity vs Subjectivity (Scatter plot)
# ----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['polarity'], df['subjectivity'], 
            c=df['polarity'], cmap='cool', s=100, alpha=0.7, edgecolors='white')
plt.colorbar(label='Polarity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.title('Sentiment Scatter Plot', color='cyan')
plt.show()

# ----------------------------
# 7️⃣ Word Cloud of social media posts
# ----------------------------
text = " ".join(df['content'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Blues').generate(text)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Social Media Posts", color='cyan')
plt.show()

# ----------------------------
# 8️⃣ Save cleaned dataset
# ----------------------------
df.to_csv("social_media_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as social_media_cleaned.csv")
