# Movie Semantic Search Assignment

This repository contains my solution for the semantic search on movie plots assignment. The project implements semantic search functionality using sentence transformers to find movies based on plot similarity.

## Project Overview

This project uses the `all-MiniLM-L6-v2` sentence transformer model to encode movie plots into embeddings and perform semantic search using cosine similarity. The solution allows users to search for movies using natural language queries that are matched against plot descriptions.

## Setup

1. **Clone the repository:**

```
git clone https://github.com/devansh-srv/AI-systems-Development
cd AI-systems-Development/movie-search-assignment
```

2. **Create and activate virtual environment:**
```
python -m venv venv
```
- On Windows:
```
venv\Scripts\activate
```

- On macOS/Linux:
```
source venv/bin/activate
```

3. **Install dependencies:**
```
pip install -r requirements.txt
```
4. **Run the script:**
```
python movie_search.py
```

## Testing

Run the unit tests to verify functionality:
```
pytest tests
```

## Usage

Test the search function with various queries:

`from movie_search import search_movies`

Example searches
```
results = search_movies('spy thriller in Paris')
results = search_movies('romantic comedy', top_n=3)
results = search_movies('space adventure')
```


## Features

- Semantic search using sentence transformers
- Cosine similarity-based ranking
- Configurable number of results (top_n parameter)
- Error handling for missing files and dependencies
- Comprehensive unit tests

## Dependencies

- sentence-transformers: For encoding text into embeddings
- pandas: For data manipulation
- scikit-learn: For cosine similarity calculations
- pytest: For testing framework

## Assignment Compliance

This solution meets all assignment requirements:
- ✅ Implements semantic search function
- ✅ Uses all-MiniLM-L6-v2 model
- ✅ Includes comprehensive unit tests
- ✅ Proper error handling
- ✅ Clean, commented code
- ✅ GitHub Actions CI/CD pipeline




