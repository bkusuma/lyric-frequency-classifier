import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_genre_distinctive_words(df, word_columns, genre_column='genre', top_n=20):
    """
    Find distinctive words for each genre using TF-IDF approach
    
    Parameters:
    df: DataFrame with genre info and word count columns
    word_columns: list of column names that contain word counts
    genre_column: name of the genre column
    top_n: number of top distinctive words per genre
    """
    
    # Step 1: Aggregate word counts by genre
    genre_word_counts = defaultdict(dict)
    
    for genre in df[genre_column].unique():
        genre_mask = df[genre_column] == genre
        genre_songs = df[genre_mask]
        
        # Sum word counts across all songs in this genre
        for word_col in word_columns:
            if word_col in df.columns:
                total_count = genre_songs[word_col].sum()
                if total_count > 0:
                    genre_word_counts[genre][word_col] = total_count
    
    # Step 2: Calculate TF-IDF-like scores for each word in each genre
    distinctive_words = {}
    
    for genre in genre_word_counts:
        word_scores = {}
        
        for word in genre_word_counts[genre]:
            # Term frequency in this genre
            tf = genre_word_counts[genre][word]
            
            # Document frequency (how many genres contain this word)
            df_count = sum(1 for g in genre_word_counts if word in genre_word_counts[g])
            
            # IDF-like score (higher when word appears in fewer genres)
            idf = np.log(len(genre_word_counts) / df_count)
            
            # TF-IDF score
            word_scores[word] = tf * idf
        
        # Get top N distinctive words for this genre
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        distinctive_words[genre] = top_words
    
    return distinctive_words

def create_genre_wordclouds(distinctive_words, cols=2):
    """
    Create word clouds for each genre's distinctive words
    """
    genres = list(distinctive_words.keys())
    rows = (len(genres) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, genre in enumerate(genres):
        # Create word frequency dict for wordcloud
        word_freq = {word.replace('_', ' '): score for word, score in distinctive_words[genre]}
        
        if word_freq:  # Only create if we have words
            wordcloud = WordCloud(
                font_path="../src/visualisation/Roboto/Roboto-VariableFont_wdth,wght.ttf",
                width=400, 
                height=300, 
                background_color=None,
                mode='RGBA',
                colormap='viridis',
                max_words=20
            ).generate_from_frequencies(word_freq)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{genre} - Distinctive Words', fontsize=14, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'{genre}\n(No distinctive words)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(genres), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Alternative approach if your data structure is different
def process_bow_dataframe(df):
    """
    Process dataframe where word counts are in separate columns
    Assumes columns follow pattern: metadata columns + word count columns
    """
    
    # Identify metadata vs word count columns
    metadata_cols = ['track_id', 'genre', 'title', 'artist_name', 'duration', 'year']
    word_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"Found {len(word_cols)} word columns")
    print(f"Genres: {df['genre'].unique()}")
    
    # Find distinctive words
    distinctive_words = analyze_genre_distinctive_words(df, word_cols)
    
    # Print top words for each genre
    for genre, words in distinctive_words.items():
        print(f"\n{genre} - Top distinctive words:")
        for word, score in words[:10]:
            print(f"  {word}: {score:.2f}")
    
    # Create word clouds
    create_genre_wordclouds(distinctive_words)
    
    return distinctive_words

# Example usage:
# distinctive_words = process_bow_dataframe(your_dataframe)

# If you want to save individual word clouds:
def save_individual_wordclouds(distinctive_words, save_path="./"):
    """Save each genre's word cloud as a separate image"""
    for genre, words in distinctive_words.items():
        word_freq = {word.replace('_', ' '): score for word, score in words}
        
        if word_freq:
            wordcloud = WordCloud(
                width=800, 
                height=600, 
                background_color='white',
                colormap='plasma',
                max_words=20
            ).generate_from_frequencies(word_freq)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'{genre} - Distinctive Words', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_path}{genre.replace('/', '_')}_wordcloud.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()

## Claude