import base64
import io
import time
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types  # Structured output ke liye zaroori hai
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TASK 1: Multimodal Image QA Schemas ---
class QARequest(BaseModel):
    image_base64: str
    question: str

# --- TASK 2: Invoice Extraction Schemas ---
class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: str
    date: str  # Format: YYYY-MM-DD
    vendor: str
    amount: float
    tax: float
    total: float

# ==================== TASK 1 ENDPOINT ====================
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
            
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            
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
            return {"answer": answer_text}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise HTTPException(status_code=500, detail=str(e))

# ==================== TASK 2 ENDPOINT ====================
@app.post("/extract")
async def extract_invoice(payload: InvoiceRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # Gemini ko instruction de rahe hain strict schema follow karne ke liye
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=payload.invoice_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceResponse,
                system_instruction=(
                    "Extract invoice data into the structured JSON format precisely.\n"
                    "Crucial Date Rule: Convert any human-readable dates (like '15 March 2026') strictly into ISO format 'YYYY-MM-DD'.\n"
                    "Crucial Numeric Rule: Extract numbers as raw floats without commas, currency strings, or extra text symbols."
                )
            ),
        )
        
        # Safe response return
        import json
        extracted_data = json.loads(response.text.strip())
        return extracted_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
