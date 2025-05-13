import os
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient as QdrantBaseClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from dotenv import load_dotenv
import logging
from models import Product
import json

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class QdrantClient:
    def __init__(self):
        """Initialize Qdrant client with configuration from environment variables."""
        logger.debug("="*50)
        logger.debug("INITIALIZING QDRANT CLIENT")
        logger.debug("="*50)
        
        # Use hardcoded values for testing
        base_url = "7bc04bf8-4c16-41c2-980a-153ec3d2aa0f.us-east-1-0.aws.cloud.qdrant.io"
        self.url = f"https://{base_url}:443"  # Explicitly use HTTPS with port 443
        self.api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.OWrnILnSEO0ctzD1r5jtaDvKlTaF4t_7a_zpJsUN11M"
        self.collection_name = "brightside-products"
        self.vector_size = 1536  # OpenAI text-embedding-3-small model dimension
        
        logger.debug("Configuration:")
        logger.debug(f"- URL: {self.url}")
        logger.debug(f"- API Key present: {'Yes' if self.api_key else 'No'}")
        logger.debug(f"- Collection name: {self.collection_name}")
        logger.debug(f"- Vector size: {self.vector_size}")
        
        # Initialize Qdrant client
        logger.debug("\nInitializing QdrantBaseClient...")
        try:
            self.client = QdrantBaseClient(
                url=self.url,
                api_key=self.api_key,
                prefer_grpc=False,  # Use HTTP/HTTPS for Qdrant Cloud
                timeout=30,  # Increase timeout for cloud connections
                https=True  # Ensure HTTPS is used
            )
            logger.debug("Successfully created QdrantBaseClient instance")
        except Exception as e:
            logger.error(f"Failed to create QdrantBaseClient: {str(e)}", exc_info=True)
            raise
        
        # Create or recreate collection with proper configuration
        logger.debug("\nSetting up collection...")
        try:
            collections = self.client.get_collections().collections
            logger.debug(f"Existing collections: {[c.name for c in collections]}")
            
            if self.collection_name in [c.name for c in collections]:
                logger.info(f"Collection '{self.collection_name}' exists, recreating...")
                self.client.delete_collection(collection_name=self.collection_name)
                logger.debug("Successfully deleted existing collection")
            
            logger.debug("Creating new collection with configuration:")
            collection_config = {
                "collection_name": self.collection_name,
                "vectors_config": {
                    "size": self.vector_size,
                    "distance": "Cosine"
                },
                "optimizers_config": {
                    "indexing_threshold": 0,
                    "memmap_threshold": 0,
                    "max_optimization_threads": 4
                },
                "on_disk_payload": True
            }
            logger.debug(f"Collection config: {json.dumps(collection_config, indent=2)}")
            
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=0,
                    max_optimization_threads=4
                ),
                quantization_config=None,
                init_from=None,
                on_disk_payload=True
            )
            logger.info(f"Successfully created collection '{self.collection_name}'")
            
            # Verify collection creation
            collection_info = self.client.get_collection(self.collection_name)
            logger.debug(f"Collection info: {json.dumps(collection_info.dict(), indent=2)}")
            
        except Exception as e:
            logger.error(f"Error in collection setup: {str(e)}", exc_info=True)
            raise
        
        logger.info(f"Successfully initialized Qdrant client for collection: {self.collection_name}")
        logger.debug("="*50)

    async def query_qdrant(
        self,
        query_vector: List[float],
        limit: int = 3,
        client_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Product]:
        """
        Query Qdrant for similar products using vector similarity search.
        """
        logger.debug("="*50)
        logger.debug("EXECUTING QDRANT QUERY")
        logger.debug("="*50)
        logger.debug(f"Query parameters:")
        logger.debug(f"- Limit: {limit}")
        logger.debug(f"- Client ID: {client_id}")
        logger.debug(f"- Filters: {json.dumps(filters, indent=2) if filters else 'None'}")
        logger.debug(f"- Query vector length: {len(query_vector)}")
        
        try:
            # Get collection info for dimension checking
            logger.debug("\nChecking collection dimensions...")
            collection_info = self.client.get_collection(self.collection_name)
            collection_dim = collection_info.config.params.vectors.size
            query_dim = len(query_vector)
            
            logger.debug(f"Dimension check:")
            logger.debug(f"- Collection dimension: {collection_dim}")
            logger.debug(f"- Query vector dimension: {query_dim}")
            
            if collection_dim != query_dim:
                error_msg = f"Vector dimension mismatch! Collection expects {collection_dim} dimensions but query vector has {query_dim} dimensions"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Build search filter
            search_filter = None
            if filters:
                logger.debug("\nBuilding search filter...")
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                logger.debug(f"Filter conditions: {json.dumps([c.dict() for c in conditions], indent=2)}")
                search_filter = Filter(must=conditions)
                logger.debug(f"Final search filter: {search_filter.dict()}")

            # Perform vector search
            logger.debug("\nExecuting vector search...")
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter
            )
            
            logger.debug(f"Found {len(search_results)} matching products")
            logger.debug(f"Search results: {json.dumps([r.dict() for r in search_results], indent=2)}")

            # Convert search results to Product models
            logger.debug("\nConverting search results to Product models...")
            products = []
            for hit in search_results:
                try:
                    logger.debug(f"\nProcessing search result:")
                    logger.debug(f"- ID: {hit.id}")
                    logger.debug(f"- Score: {hit.score}")
                    logger.debug(f"- Payload: {json.dumps(hit.payload, indent=2)}")
                    
                    # Get required fields with fallbacks
                    title = hit.payload.get("title", "")
                    description = hit.payload.get("description", "")
                    price = hit.payload.get("price", "0.00")
                    image = hit.payload.get("image", "")
                    link = hit.payload.get("link", "")
                    
                    # Create product
                    product = Product(
                        id=str(hit.payload.get("id", "")),
                        name=title,
                        description=description,
                        price=price,
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
                    logger.debug(f"Successfully created product: {product.dict()}")
                except Exception as e:
                    logger.error(f"Error processing product {hit.id}: {str(e)}", exc_info=True)
                    logger.error(f"Problematic payload: {json.dumps(hit.payload, indent=2)}")
                    continue

            logger.debug(f"\nQuery completed successfully. Returning {len(products)} products.")
            logger.debug("="*50)
            return products

        except Exception as e:
            logger.error(f"Error in query_qdrant: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    async def close(self):
        """Close the Qdrant client connection."""
        logger.debug("="*50)
        logger.debug("CLOSING QDRANT CLIENT")
        logger.debug("="*50)
        try:
            self.client.close()
            logger.debug("Successfully closed Qdrant client connection")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {str(e)}", exc_info=True)
            raise
        logger.debug("="*50) 
