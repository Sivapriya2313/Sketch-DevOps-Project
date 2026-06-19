from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn.functional as F
import io
import os
import glob
import torchvision.transforms as transforms

# Import your actual Phase 2 model class 
from model_phase2 import Phase2ResNet50 

app = FastAPI(title="Sketch Retrieval API")

print("Setting up the High-Performance Sketch Engine...")

# =====================================================================
# 🧠 HARDWARE & MODEL SETUP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on hardware acceleration device: {device}")

model = Phase2ResNet50() 
weights_path = "phase2_weights_chair/ckpt_epoch_80.pth" 

if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval() 
    print("🧠 Thesis PyTorch model weights loaded successfully!")
else:
    print(f"⚠️ CRITICAL ERROR: Weights file missing!")

# =====================================================================
# 🗄️ RUNTIME DATABASE INDEXING
# =====================================================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# This dictionary will hold all our dataset math vectors in RAM
database_embeddings = {}
dataset_dir = "/app/trainB"

@app.on_event("startup")
async def build_database():
    """
    When the server boots up, it looks at the mounted Docker folder,
    processes every image, and saves its mathematical vector into memory.
    """
    print(f"🔍 Looking for dataset images in: {dataset_dir}")
    
    # Grab all PNG and JPG files
    image_paths = glob.glob(os.path.join(dataset_dir, "*.png")) + glob.glob(os.path.join(dataset_dir, "*.jpg"))
    
    if len(image_paths) == 0:
        print("⚠️ Warning: No images found in the dataset folder!")
        return

    print(f"⚙️ Extracting features for {len(image_paths)} images. This might take a minute...")
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                
                # Get the embedding (we ignore the abs_level for the database)
                emb, _ = model(input_tensor)
                
                # Save the flat math vector to our dictionary using the filename
                filename = os.path.basename(img_path)
                database_embeddings[filename] = emb.squeeze(0)
            except Exception as e:
                pass # Skip broken images
                
    print(f"✅ Successfully indexed {len(database_embeddings)} real images into active memory!")

#=====================================================================
# 🎯 THE PREDICTION ENDPOINT
# =====================================================================
@app.post("/predict")
async def predict_sketch(file: UploadFile = File(...)):
    # 1. Read the uploaded sketch
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device) 
    
    # 2. Extract the sketch's mathematical features
    with torch.no_grad():
        sketch_embedding, _ = model(input_tensor)
        sketch_emb_flat = sketch_embedding.squeeze(0)
    
    # 3. Compare the sketch to EVERY image in our database dictionary
    results = []
    for filename, db_embedding in database_embeddings.items():
        # We use Cosine Similarity to see how closely the vectors match
        similarity = F.cosine_similarity(sketch_emb_flat.unsqueeze(0), db_embedding.unsqueeze(0))
        results.append((filename, similarity.item()))
    
    # 4. Sort the list by highest similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 👇 THIS IS EXACTLY WHERE YOUR NEW CODE GOES 👇
    # =====================================================================
    # 1. Get the math score of the closest match
    best_match_score = results[0][1] 
    
    # I added a print statement here so you can see the exact math score in your terminal!
    print(f"Top match similarity score: {best_match_score:.4f}") 
    
    # 2. Define the exact cutoff point (You must test and change this number!)
    MINIMUM_CHAIR_SCORE = 0.85 

    # 3. THE LOGIC: If the score is too low, it's probably a shoe or random object. Reject it.
    if best_match_score < MINIMUM_CHAIR_SCORE:
        return {
            "status": "error",
            "message": "⚠️ Error: Object not recognized. Please upload a valid chair sketch."
        }
        
    # 4. If it passes the test, it's a valid chair! Return the results.
    top_10_matches = [res[0] for res in results[:10]]
    return {
        "status": "success",
        "matches": top_10_matches
    }
    # =====================================================================