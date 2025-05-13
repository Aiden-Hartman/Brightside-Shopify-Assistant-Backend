import os
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient as QdrantBaseClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import logging
from models import Product

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class QdrantClient:
    def __init__(self):
        """Initialize Qdrant client with configuration from environment variables."""
        logger.debug("Initializing QdrantClient...")
        
        # Use hardcoded values for testing
        base_url = "7bc04bf8-4c16-41c2-980a-153ec3d2aa0f.us-east-1-0.aws.cloud.qdrant.io"
        self.url = f"https://{base_url}:443"  # Explicitly use HTTPS with port 443
        self.api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.OWrnILnSEO0ctzD1r5jtaDvKlTaF4t_7a_zpJsUN11M"
        self.collection_name = "brightside-products"
        self.vector_size = 1536  # OpenAI text-embedding-3-small model dimension
        
        logger.debug(f"Using Qdrant Cloud URL: {self.url}")
        logger.debug(f"API Key present: {'Yes' if self.api_key else 'No'}")
        logger.debug(f"Collection name: {self.collection_name}")
        logger.debug(f"Vector size: {self.vector_size}")
        
        # Initialize Qdrant client
        logger.debug("Creating QdrantBaseClient...")
        self.client = QdrantBaseClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=False,  # Use HTTP/HTTPS for Qdrant Cloud
            timeout=30,  # Increase timeout for cloud connections
            https=True  # Ensure HTTPS is used
        )
        logger.info(f"Initialized Qdrant client for collection: {self.collection_name}")
        logger.debug(f"[DEBUG] Connected to Qdrant Cloud at {self.url}")

    async def query_qdrant(
        self,
        query_vector: List[float],
        limit: int = 3,
        client_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Product]:
        """
        Query Qdrant for similar products using vector similarity search.
        
        Args:
            query_vector: Vector representation of the search query
            limit: Maximum number of results to return
            client_id: Optional client ID (not used for filtering)
            filters: Optional additional metadata filters
            
        Returns:
            List[Product]: List of matching products
        """
        try:
            # Get collection info for dimension checking
            collection_info = self.client.get_collection(self.collection_name)
            collection_dim = collection_info.config.params.vectors.size
            query_dim = len(query_vector)
            
            logger.debug(f"Vector dimension check:")
            logger.debug(f"- Collection dimension: {collection_dim}")
            logger.debug(f"- Query vector dimension: {query_dim}")
            
            if collection_dim != query_dim:
                error_msg = f"Vector dimension mismatch! Collection expects {collection_dim} dimensions but query vector has {query_dim} dimensions"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Query parameters - limit: {limit}, filters: {filters}")
            
            # Build search filter (without client_id)
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                logger.debug(f"Added additional filters: {filters}")
                search_filter = Filter(must=conditions)
                logger.debug(f"Final search filter: {search_filter}")

            # Log that we're performing a global search
            logger.info("[Qdrant] Performing global product search without client_id filter.")

            # Perform vector search
            logger.debug("Executing Qdrant search...")
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter
            )
            
            logger.debug(f"Found {len(search_results)} matching products")

            # Convert search results to Product models
            products = []
            for hit in search_results:
                try:
                    logger.debug(f"Processing search result - ID: {hit.id}, Score: {hit.score}")
                    logger.debug(f"Raw payload: {hit.payload}")  # Add debug logging for payload
                    
                    # Get required fields with fallbacks
                    title = hit.payload.get("title", "")
                    description = hit.payload.get("description", "")
                    price = hit.payload.get("price", "0.00")  # Already a string from Qdrant
                    image = hit.payload.get("image", "")
                    link = hit.payload.get("link", "")
                    
                    # Create product with required fields
                    product = Product(
                        id=str(hit.payload.get("id", "")),
                        name=title,
                        description=description,
                        price=price,  # Price is already a string
                        currency="USD",
                        image_url=image,
                        product_url=link,
                        score=hit.score,
                        brand="Brightside",
                        category=None,
                        tags=None,
                        variants=None,
                        ingredients=None,
                        nutritional_info=None,
                        allergens=None,
                        dietary_info=None,
                        rating=None,
                        review_count=None,
                        metadata=None
                    )
                    products.append(product)
                    logger.debug(f"Successfully processed product: {product.name} (score: {hit.score:.3f})")
                except Exception as e:
                    logger.error(f"Error parsing product {hit.id}: {str(e)}", exc_info=True)
                    logger.error(f"Payload that caused error: {hit.payload}")  # Add error logging for payload
                    continue

            return products

        except Exception as e:
            logger.error(f"Error querying Qdrant: {str(e)}", exc_info=True)
            raise

    async def close(self):
        """Close the Qdrant client connection."""
        logger.debug("Closing Qdrant client connection...")
        self.client.close() 