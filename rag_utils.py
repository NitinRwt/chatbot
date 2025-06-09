import os
import json
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict, Any, Tuple

# Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Error: Set GOOGLE_API_KEY environment variable before running.")
genai.configure(api_key=API_KEY)

# File paths
DATA_DIR = "data"
PROFILE_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings_profiles.jsonl")
JOB_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings_jobs.jsonl")

# Global variables to cache loaded data
_profile_data = None
_job_data = None
_profile_index = None
_job_index = None

def load_embeddings_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load embeddings data from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing embeddings
        
    Returns:
        List of dictionaries containing id, text, and embedding
    """
    if not os.path.exists(file_path):
        print(f"Warning: Embeddings file not found: {file_path}")
        return []
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    if 'embedding' in record and 'text' in record and 'id' in record:
                        data.append(record)
                    else:
                        print(f"Warning: Missing required fields in line {line_num} of {file_path}")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error in line {line_num} of {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return data

def build_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        FAISS index for similarity search
    """
    if not embeddings:
        return None
    
    # Convert to numpy array
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    
    # Get dimension
    dimension = embedding_matrix.shape[1]
    
    # Create FAISS index (using L2 distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embedding_matrix)
    
    return index

def get_query_embedding(query: str, model: str = "models/text-embedding-004") -> List[float]:
    """
    Get embedding for a query string using Gemini.
    
    Args:
        query: Query string to embed
        model: Embedding model to use
        
    Returns:
        Embedding vector as list of floats
    """
    try:
        response = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query"  # Use query task type for search queries
        )
        return response['embedding']
    except Exception as e:
        print(f"Error getting query embedding: {e}")
        # Return a zero vector as fallback (this won't give good results but prevents crashes)
        return [0.0] * 768  # Assuming 768-dimensional embeddings

def initialize_profile_data():
    """Initialize profile data and FAISS index if not already loaded."""
    global _profile_data, _profile_index
    
    if _profile_data is None:
        print("Loading profile embeddings...")
        _profile_data = load_embeddings_data(PROFILE_EMBEDDINGS)
        
        if _profile_data:
            embeddings = [record['embedding'] for record in _profile_data]
            _profile_index = build_faiss_index(embeddings)
            print(f"Loaded {len(_profile_data)} profile embeddings")
        else:
            print("No profile embeddings found")
            _profile_index = None

def initialize_job_data():
    """Initialize job data and FAISS index if not already loaded."""
    global _job_data, _job_index
    
    if _job_data is None:
        print("Loading job embeddings...")
        _job_data = load_embeddings_data(JOB_EMBEDDINGS)
        
        if _job_data:
            embeddings = [record['embedding'] for record in _job_data]
            _job_index = build_faiss_index(embeddings)
            print(f"Loaded {len(_job_data)} job embeddings")
        else:
            print("No job embeddings found")
            _job_index = None

def retrieve_profiles(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most similar profiles based on a query.
    
    Args:
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing profile information
    """
    # Initialize data if needed
    initialize_profile_data()
    
    if not _profile_data or _profile_index is None:
        print("No profile data available for search")
        return []
    
    # Get query embedding
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return []
    
    # Search using FAISS
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    try:
        # Search for top_k similar profiles
        distances, indices = _profile_index.search(query_vector, min(top_k, len(_profile_data)))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_profile_data):  # Ensure valid index
                profile = _profile_data[idx].copy()
                profile['similarity_score'] = float(distances[0][i])  # Lower is better for L2 distance
                results.append(profile)
        
        return results
        
    except Exception as e:
        print(f"Error during profile search: {e}")
        return []

def retrieve_jobs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most similar job listings based on a query.
    
    Args:
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing job information
    """
    # Initialize data if needed
    initialize_job_data()
    
    if not _job_data or _job_index is None:
        print("No job data available for search")
        return []
    
    # Get query embedding
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        return []
    
    # Search using FAISS
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    try:
        # Search for top_k similar jobs
        distances, indices = _job_index.search(query_vector, min(top_k, len(_job_data)))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_job_data):  # Ensure valid index
                job = _job_data[idx].copy()
                job['similarity_score'] = float(distances[0][i])  # Lower is better for L2 distance
                results.append(job)
        
        return results
        
    except Exception as e:
        print(f"Error during job search: {e}")
        return []

def search_profiles_by_keywords(keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search profiles using keyword matching as a fallback method.
    
    Args:
        keywords: List of keywords to search for
        top_k: Number of top results to return
        
    Returns:
        List of matching profiles
    """
    initialize_profile_data()
    
    if not _profile_data:
        return []
    
    results = []
    keywords_lower = [kw.lower() for kw in keywords]
    
    for profile in _profile_data:
        text_lower = profile['text'].lower()
        score = sum(1 for kw in keywords_lower if kw in text_lower)
        
        if score > 0:
            profile_copy = profile.copy()
            profile_copy['keyword_score'] = score
            results.append(profile_copy)
    
    # Sort by keyword score (descending)
    results.sort(key=lambda x: x['keyword_score'], reverse=True)
    
    return results[:top_k]

def search_jobs_by_keywords(keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search jobs using keyword matching as a fallback method.
    
    Args:
        keywords: List of keywords to search for
        top_k: Number of top results to return
        
    Returns:
        List of matching jobs
    """
    initialize_job_data()
    
    if not _job_data:
        return []
    
    results = []
    keywords_lower = [kw.lower() for kw in keywords]
    
    for job in _job_data:
        text_lower = job['text'].lower()
        score = sum(1 for kw in keywords_lower if kw in text_lower)
        
        if score > 0:
            job_copy = job.copy()
            job_copy['keyword_score'] = score
            results.append(job_copy)
    
    # Sort by keyword score (descending)
    results.sort(key=lambda x: x['keyword_score'], reverse=True)
    
    return results[:top_k]

def get_stats():
    """Get statistics about loaded data."""
    initialize_profile_data()
    initialize_job_data()
    
    profile_count = len(_profile_data) if _profile_data else 0
    job_count = len(_job_data) if _job_data else 0
    
    return {
        "profiles_loaded": profile_count,
        "jobs_loaded": job_count,
        "profile_index_ready": _profile_index is not None,
        "job_index_ready": _job_index is not None
    }

# Test functions
def test_profile_search():
    """Test profile search functionality."""
    test_queries = [
        "Python developer",
        "React frontend engineer",
        "Data scientist with machine learning",
        "Remote backend developer"
    ]
    
    print("Testing profile search...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retrieve_profiles(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text'][:100]}...")

def test_job_search():
    """Test job search functionality."""
    test_queries = [
        "Remote React developer",
        "Python backend engineer",
        "Data science position",
        "Full stack developer"
    ]
    
    print("Testing job search...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retrieve_jobs(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text'][:100]}...")

if __name__ == "__main__":
    # Print stats
    stats = get_stats()
    print("RAG Utils Statistics:")
    print(f"  Profiles loaded: {stats['profiles_loaded']}")
    print(f"  Jobs loaded: {stats['jobs_loaded']}")
    print(f"  Profile index ready: {stats['profile_index_ready']}")
    print(f"  Job index ready: {stats['job_index_ready']}")
    
    # Run tests if data is available
    if stats['profiles_loaded'] > 0:
        test_profile_search()
    
    if stats['jobs_loaded'] > 0:
        test_job_search()