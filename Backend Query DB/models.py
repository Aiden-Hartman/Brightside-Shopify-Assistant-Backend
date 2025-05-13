from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from decimal import Decimal
from datetime import datetime

class RecommendRequest(BaseModel):
    """Request model for product recommendations."""
    query: str = Field(..., description="The user's search query")
    limit: int = Field(default=3, description="Maximum number of products to return")
    client_id: Optional[str] = Field(default=None, description="Optional client ID filter")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional additional metadata filters")

class ShippingInfo(BaseModel):
    """Shipping information for a product variant."""
    weight: Optional[float] = Field(None, description="Weight in grams")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Dimensions in cm (length, width, height)")
    free_shipping: bool = Field(default=False, description="Whether this variant qualifies for free shipping")

class ProductVariant(BaseModel):
    """Product variant information."""
    id: str = Field(..., description="Unique identifier for the variant")
    sku: Optional[str] = Field(None, description="Stock keeping unit")
    price: Decimal = Field(..., description="Price of the variant")
    currency: str = Field(..., description="Currency code (e.g., USD)")
    in_stock: bool = Field(default=True, description="Whether the variant is in stock")
    shipping_info: Optional[ShippingInfo] = Field(None, description="Shipping information for this variant")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional variant attributes (color, size, etc.)")

class Product(BaseModel):
    """Product recommendation result."""
    # Required fields
    id: str = Field(..., description="Unique product identifier")
    name: str = Field(default="", description="Product name")
    description: str = Field(default="", description="Product description")
    price: str = Field(default="0.00", description="Base price of the product")
    currency: str = Field(default="USD", description="Currency code (e.g., USD)")
    image_url: str = Field(default="", description="URL to the product image")
    product_url: str = Field(default="", description="URL to the product page")
    
    # Optional fields
    score: Optional[float] = Field(None, description="Relevance score from vector search")
    variants: Optional[List[ProductVariant]] = Field(None, description="Available product variants")
    brand: Optional[str] = Field(None, description="Product brand name")
    category: Optional[str] = Field(None, description="Product category")
    tags: Optional[List[str]] = Field(None, description="Product tags")
    ingredients: Optional[List[str]] = Field(None, description="Product ingredients")
    nutritional_info: Optional[Dict[str, Any]] = Field(None, description="Nutritional information")
    allergens: Optional[List[str]] = Field(None, description="Allergen information")
    dietary_info: Optional[Dict[str, bool]] = Field(None, description="Dietary information (vegan, gluten-free, etc.)")
    rating: Optional[float] = Field(None, description="Product rating (0-5)")
    review_count: Optional[int] = Field(None, description="Number of reviews")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional product metadata")

class RecommendResponse(BaseModel):
    """Response model for product recommendations."""
    type: str = Field(default="recommendation", description="Response type")
    query: str = Field(..., description="Original search query")
    products: List[Product] = Field(..., description="List of recommended products")
    total: int = Field(..., description="Total number of products returned")

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str = Field(..., description="The role of the message sender (user/assistant)")
    content: str = Field(..., description="The content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Message timestamp")

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="The user's input message")
    client_id: Optional[str] = Field(default=None, description="Optional client ID for tracking")
    session_id: Optional[str] = Field(default=None, description="Optional session ID for grouping messages")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Optional previous messages in the conversation")
    quiz_answers: Optional[Dict[str, Any]] = Field(default=None, description="Optional quiz answers containing user preferences, budget, and goals")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    role: str = Field(default="assistant", description="The role of the message sender")
    content: str = Field(..., description="The assistant's response message")
    recommend: Optional[bool] = Field(default=None, description="Whether to recommend products based on the response")
    products: Optional[List[Product]] = Field(default=None, description="List of recommended products if any")
    function_called: Optional[bool] = Field(default=False, description="Whether a function was called in this response")
    function_name: Optional[str] = Field(default=None, description="Name of the function that was called, if any") 