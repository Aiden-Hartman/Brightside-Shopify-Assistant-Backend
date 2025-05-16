from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import Dict, Any
import logging

from models import RecommendRequest, RecommendResponse, Product
from embedder import Embedder
from my_qdrant_utils import QdrantClient
from utils import log_error, log_info
from chat_route import router as chat_router

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Recommendation API")

# Enable CORS for localhost development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://brightside-shopify-assistant-frontend.vercel.app"  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
logger.debug("Initializing Embedder...")
embedder = Embedder()  # API key is loaded from environment variables
logger.debug("Initializing QdrantClient...")
qdrant = QdrantClient()

# Include the chat router
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Mock products for fallback
mockProducts = [
    {
        "id": "1",
        "title": "Organic Fresh Avocados",
        "price": "4.99",
        "description": "Perfectly ripe, hand-picked Hass avocados. Rich, creamy, and perfect for any meal.",
        "image": "https://images.unsplash.com/photo-1523049673857-eb18f1d7b578",
        "link": "/products/organic-avocados"
    },
    {
        "id": "2",
        "title": "Premium Greek Yogurt",
        "price": "3.49",
        "description": "Creamy, protein-rich Greek yogurt. Made with all-natural ingredients.",
        "image": "https://images.unsplash.com/photo-1488477181946-6428a0291777",
        "link": "/products/greek-yogurt"
    },
    {
        "id": "3",
        "title": "Artisan Sourdough Bread",
        "price": "6.99",
        "description": "Freshly baked sourdough bread with a perfect crust and chewy interior.",
        "image": "https://images.unsplash.com/photo-1509440159596-0249088772ff",
        "link": "/products/sourdough-bread"
    },
    {
        "id": "4",
        "title": "Organic Wild Blueberries",
        "price": "5.99",
        "description": "Sweet and tangy wild blueberries, perfect for snacking or baking.",
        "image": "https://images.unsplash.com/photo-1498557850523-fd3d118b962e",
        "link": "/products/wild-blueberries"
    }
]

@app.get("/health")
async def health_check() -> Dict[str, str]:
    logger.debug("Health check endpoint called")
    return {"status": "ok"}

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    try:
        # Log the received request
        logger.debug(f"Received recommendation request: {request.dict()}")

        # Get embedding for the query
        start_time = time.time()
        logger.debug("Calling embedder.embed_text...")
        query_vector = await embedder.embed_text(request.query)
        embedding_time = time.time() - start_time
        logger.debug(f"Generated embedding in {embedding_time:.2f}s (vector size: {len(query_vector)})")

        # Query Qdrant
        logger.debug("Calling qdrant.query_qdrant...")
        products = await qdrant.query_qdrant(
            query_vector=query_vector,
            limit=request.limit,
            client_id=request.client_id,
            filters=request.filters
        )
        logger.debug(f"Retrieved {len(products)} products from Qdrant")

        # If no products found, use mock products
        if not products:
            logger.info("No products found in Qdrant, using mock products")
            products = mockProducts
        else:
            # Transform products to match frontend expectations
            products = [
                {
                    "id": str(p.id),
                    "title": p.name,
                    "price": str(p.price),
                    "description": p.description,
                    "image": p.image_url,
                    "link": p.product_url,
                    "formattedPrice": f"${p.price}",
                    "score": p.score
                }
                for p in products
            ]

        # Add debug log before returning
        logger.info(f"[Backend] Returning products: {products}")
        
        return products

    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}", exc_info=True)
        # Return mock products on error
        logger.info("Error occurred, returning mock products")
        return mockProducts

@app.on_event("shutdown")
async def shutdown_event():
    logger.debug("Shutting down application...")
    await embedder.close()
    await qdrant.close() 
