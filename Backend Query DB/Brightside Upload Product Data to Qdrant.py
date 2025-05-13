import json
import numpy as np
from tqdm import tqdm
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# --- Configuration ---
COLLECTION_NAME = "brightside-products"
JSON_FILE_PATH = r"C:\Users\aiden\.anaconda\Brightside\brightside_product_data_completed.json"
QDRANT_URL = "https://7bc04bf8-4c16-41c2-980a-153ec3d2aa0f.us-east-1-0.aws.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.OWrnILnSEO0ctzD1r5jtaDvKlTaF4t_7a_zpJsUN11M"
VECTOR_SIZE = 1536  # Size for OpenAI text-embedding-3-small model

# --- Initialize client ---
print("Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=API_KEY,
    prefer_grpc=False,  # Use HTTP/HTTPS for Qdrant Cloud
    timeout=30,
)

# --- Load the JSON ---
print("Loading product data...")
with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)  # data is a flat list of product dicts

# --- Create collection if it doesn't exist ---
print("Creating/recreating collection...")
collections = client.get_collections().collections
if COLLECTION_NAME in [c.name for c in collections]:
    print(f"Collection '{COLLECTION_NAME}' exists, recreating...")
    client.delete_collection(collection_name=COLLECTION_NAME)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE  # Cosine similarity is better for normalized vectors
    ),
)

# --- Process products and create points ---
print("Generating points with random embeddings...")
all_points = []
for product in tqdm(data, desc="Processing products"):  # data is a list
    # Generate a random vector and normalize it
    vector = np.random.normal(size=VECTOR_SIZE)
    vector = vector / np.linalg.norm(vector)  # Normalize to unit length
    
    # Create a point for each product with its full information
    point = PointStruct(
        id=str(uuid4()),
        vector=vector.tolist(),  # Convert numpy array to list
        payload={
            "id": str(product.get("id")),  # Ensure ID is string
            "title": product.get("title", ""),
            "description": product.get("description", ""),
            "price": str(product.get("price", "0.00")),  # Ensure price is string
            "formattedPrice": product.get("formattedPrice", ""),
            "image": product.get("image", ""),
            "link": product.get("link", "")
        }
    )
    all_points.append(point)

# --- Upload in batches with progress bar ---
print("\nUploading points to Qdrant...")
BATCH_SIZE = 100
for i in tqdm(range(0, len(all_points), BATCH_SIZE), desc="Uploading batches"):
    batch = all_points[i:i+BATCH_SIZE]
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=batch,
        wait=True  # block until write completes
    )

print(f"\n✅ Successfully uploaded {len(all_points)} points to '{COLLECTION_NAME}' collection")
print(f"Collection configuration:")
print(f"- Vector size: {VECTOR_SIZE}")
print(f"- Distance metric: Cosine similarity")
print(f"- Number of products: {len(all_points)}") 