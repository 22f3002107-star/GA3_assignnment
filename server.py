import base64
import io
import time
import os  # <-- Yeh zaroori hai Render variables read karne ke liye
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: QARequest):
    max_retries = 3
    delay = 2
    
    for attempt in range(max_retries):
        try:
            img_str = payload.image_base64
            if "," in img_str:
                img_str = img_str.split(",")[-1]
            missing_padding = len(img_str) % 4
            if missing_padding:
                img_str += '=' * (4 - missing_padding)
            image_bytes = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(image_bytes))
            
            # --- API Key securely fetched from Render Environment ---
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            # --------------------------------------------------------
            
            prompt = (
                f"Question: {payload.question}\n\n"
                "Task: Answer the question directly based on the image.\n"
                "Strict Rule for numbers: If the answer is a numeric value, output ONLY the raw number digits (e.g., 4089.35). Do not include any words, commas, letters, currency symbols, or units."
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, image]
            )
            answer_text = response.text.strip()
            
            print(f"\n[SUCCESS] Q: {payload.question} -> A: {answer_text}")
            return {"answer": answer_text}
            
        except Exception as e:
            print(f"[ATTEMPT {attempt + 1} FAILED]: Retrying due to network/demand load...")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  
            else:
                import traceback
                print(f"\n[FINAL SERVER EXCEPTION]:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
