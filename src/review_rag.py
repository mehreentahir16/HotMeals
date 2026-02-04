"""
Review RAG interface for BiteBot.

Provides semantic search over restaurant reviews stored in Pinecone.
"""

import os
import logging
from typing import Optional, List, Dict
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
_openai_client = None
_pinecone_index = None
INDEX_NAME = "bitebot-reviews"
EMBEDDING_MODEL = "text-embedding-3-small"


def _get_openai_client():
    """Lazy initialization of OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_pinecone_index():
    """Lazy initialization of Pinecone index."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text query."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def search_reviews(
    business_id: str,
    query: Optional[str] = None,
    top_k: int = 10,
    min_stars: Optional[float] = None,
) -> List[Dict]:
    """
    Search for restaurant reviews.
    
    Args:
        business_id: Yelp business ID to filter reviews
        query: Optional semantic search query (e.g. "service", "noisy", "romantic")
        top_k: Number of reviews to return
        min_stars: Optional minimum star rating filter
        
    Returns:
        List of review dicts with keys: review_id, text, stars, date, useful, score
    """
    logger.info(f"[REVIEW_RAG] Searching reviews for business_id={business_id}, query={query}")
    
    try:
        index = _get_pinecone_index()
        
        # Build metadata filter
        filter_dict = {"business_id": {"$eq": business_id}}
        if min_stars is not None:
            filter_dict["stars"] = {"$gte": min_stars}
        
        if query:
            # Semantic search
            query_embedding = generate_embedding(query)
            results = index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
        else:
            # No query → just get top reviews by metadata (sorted by usefulness)
            # Pinecone doesn't support pure metadata queries without a vector,
            # so we'll do a dummy query with a zero vector (this returns results
            # sorted by metadata matching)
            results = index.query(
                vector=[0.0] * 1536,  # dummy vector
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
        
        # Format results
        reviews = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            reviews.append({
                'review_id': match['id'],
                'text': metadata.get('text', ''),
                'stars': metadata.get('stars', 0),
                'date': metadata.get('date', ''),
                'useful': metadata.get('useful', 0),
                'score': match.get('score', 0.0),  # similarity score if query provided
            })
        
        logger.info(f"[REVIEW_RAG] Found {len(reviews)} reviews")
        return reviews
        
    except Exception as e:
        logger.error(f"[REVIEW_RAG] Error: {e}", exc_info=True)
        return []


def get_review_summary(business_id: str) -> Dict:
    """
    Get aggregate review statistics for a restaurant.
    
    Args:
        business_id: Yelp business ID
        
    Returns:
        Dict with keys: avg_stars, total_reviews, recent_reviews
    """
    # This would require aggregating over all reviews for a business.
    # Pinecone doesn't do aggregations natively, so for MVP we can skip this
    # or implement it by fetching all reviews and computing locally.
    # For now, return a placeholder.
    return {
        'avg_stars': None,
        'total_reviews': None,
        'recent_reviews': search_reviews(business_id, top_k=5)
    }