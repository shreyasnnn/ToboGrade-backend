#uvicorn app:app --reload
import os
import uuid
from datetime import datetime
from mangum import Mangum
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import tempfile
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Config
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("✅ Supabase client created successfully")

# Load your 4-grade model
try:
    model = load_model("mv2_tobacco_model.h5")  # Your new 4-class model
    logger.info("✅ 4-grade tobacco model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise

app = FastAPI(
    title="ToboGradre API - 4 Grade System",
    description="Tobacco Leaf Quality Detection API (4 Grades)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://leafgrade-tobacco-grading-system.netlify.app/",  # Your Netlify domain
        "http://localhost:5173",                # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated class list for 4 grades
TOBACCO_CLASSES = ['Grade_1', 'Grade_4', 'Grade_5', 'RedThargu']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        logger.info(f"📤 Running 4-grade prediction for: {file.filename}")

        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        file_bytes = await file.read()
        filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # ✅ Fixed: Preprocess image to match model input shape
        img = image.load_img(temp_path, target_size=(640, 640))  # Changed from (224, 224)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict using 4-class model
        preds = model.predict(img_array)
        confidence = float(np.max(preds) * 100)
        class_index = int(np.argmax(preds))
        result = TOBACCO_CLASSES[class_index]

        logger.info(f"🎯 4-Grade Prediction: {result} ({confidence:.2f}%)")

        return {
            "result": result,
            "confidence": round(confidence, 2),
            "message": "4-grade prediction completed successfully",
            "available_grades": TOBACCO_CLASSES,
            "total_classes": len(TOBACCO_CLASSES)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/save")
async def save_to_history(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    result: str = Form(...),
    confidence: float = Form(...)
):
    try:
        logger.info(f"💾 Saving 4-grade result for user {user_id}")

        # Validate result is one of the 4 grades
        if result not in TOBACCO_CLASSES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid grade. Must be one of: {TOBACCO_CLASSES}"
            )

        file_bytes = await file.read()
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        safe_filename = f"{timestamp}_{uuid.uuid4()}_{file.filename}"
        folder_path = f"user/{user_id}/images/{safe_filename}"

        # Upload to Supabase storage
        upload_response = supabase.storage.from_(BUCKET_NAME).upload(
            path=folder_path,
            file=file_bytes,
            file_options={"content-type": file.content_type}
        )

        if hasattr(upload_response, "error") and upload_response.error:
            raise Exception(f"Storage upload failed: {upload_response.error}")

        image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{folder_path}"

        # Insert record into DB
        insert_data = {
            "user_id": user_id,
            "image_url": image_url,
            "result": result,
            "confidence": str(confidence),
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat(),
            "model_version": "4-grade-v1.0"  # Track model version
        }
        
        insert_response = supabase.table("upload_history").insert(insert_data).execute()

        if hasattr(insert_response, "error") and insert_response.error:
            raise Exception(f"Database insert failed: {insert_response.error}")

        logger.info(f"✅ 4-grade result saved successfully")
        return {
            "message": "Saved to history successfully",
            "image_url": image_url,
            "path": folder_path,
            "grade": result,
            "model_version": "4-grade-v1.0"
        }

    except Exception as e:
        logger.error(f"❌ Save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "available_grades": TOBACCO_CLASSES,
        "total_classes": len(TOBACCO_CLASSES),
        "model_version": "4-grade-v1.0",
        "tensorflow_version": "2.15.0"
    }

@app.get("/grades")
async def get_available_grades():
    """Get currently supported tobacco grades"""
    return {
        "current_grades": TOBACCO_CLASSES,
        "total_current": len(TOBACCO_CLASSES),
        "future_grades": ["BlackThargu", "SemiGreen", "Bright_green"],
        "total_planned": 7,
        "status": "4-grade system active"
    }

handler = Mangum(app)
