import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and create embeddings (global for testing)
"""
The dataset contaings 2 columns
>> df.columns
>> Index(['title', 'plot'], dtype='object')
we need to create embeddings for this using all-MiniLM-L6-v2  
"""
try:
    df = pd.read_csv("./movies.csv")
except Exception as e:
    df = None
    print(f"Error in loading dataset: {e}")

# Load the Sentence Transformer model
"""
all-MiniLM-L6-v2 can be loaded using SentenceTransformer class from the sentence_transformers module
"""
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    model = None
    print(f"Error in loading model {e}")


# Convert the 'plot of the movies into an embedding
"""
.encode() method can be used for encoding the plots
we just need to encode `plot` column using df['plot']
"""
if model is not None and  df is not None and 'plot' in df.columns:
    try:
        embeddings = model.encode(df['plot'].tolist(),convert_to_tensor=False)
    except Exception as e:
        print(f"Error encoding plots: {e}")
else:
    if model is None:
        print("Model `model` is not properly defined")
    if df is None:
        print("DataFrame `df` is not properly defined or does not contain the `plot` column")

def search_movies(query, top_n=5):
    #TO_DO logic for implementing
    """
    Search for movies based on semantic similarity to the query.
    
    Args:
        query (str): Search query describing the type of movie
        top_n (int): Number of top results to return (default: 5)
    
    Returns:
        pd.DataFrame: DataFrame with columns ['title', 'plot', 'similarity']
                     sorted by similarity in descending order, or None if error
    
    Example:
        >>> results = search_movies('spy thriller in Paris')
        >>> print(results[['title', 'similarity']].head())
    """
    if model is not None and query is not None and df is not None:
        query_embeddings = model.encode(query,convert_to_tensor=False)
        similarities = cosine_similarity([query_embeddings],embeddings)[0]
        results = pd.DataFrame({
            'title':df['title'],
            'plot':df['plot'],
            'similarity': similarities
        })
        results = results.sort_values(by='similarity',ascending=False)
        return results.head(top_n)

    else:
        if model is None:
            print("Model `model` is not properly defined")
        if query is None:
            print("Query is not provided.")
        if df is None:
            print("DataFrame `df` is not properly defined or does not contain the `plot` column")

    return None


