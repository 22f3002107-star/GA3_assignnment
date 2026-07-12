import base64
import io
import os
import json
import asyncio
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types  
from PIL import Image
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")
client = genai.Client(api_key=API_KEY)

RATE_LIMIT_LOCK = asyncio.Lock()

class QARequest(BaseModel):
    image_base64: str
    question: str

class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None  
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None  

def decode_image_helper(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[-1]
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))

# ==================== TASK 1: MULTIMODAL QA ====================
@app.post("/answer-image")
async def answer_image(payload: QARequest):
    async with RATE_LIMIT_LOCK:
        max_retries = 4
        delay = 1.5
        
        for attempt in range(max_retries):
            try:
                image = decode_image_helper(payload.image_base64)
                
                prompt = (
                    f"Question: {payload.question}\n\n"
                    "Task: Answer the question directly based on the provided image.\n"
                    "Strict Rule for numbers: If the answer is a numeric value, output ONLY the raw number digits (e.g., 4089.35). Do not include any words, commas, letters, currency symbols, or units.\n"
                    "Strict Rule for text: If the answer is text, output just the direct answer plainly without extra conversational text."
                )
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image]
                )
                
                if not response.text:
                    raise Exception("Empty response from Gemini API")
                
                await asyncio.sleep(1.0)
                return {"answer": response.text.strip()}

            except Exception as e:
                print(f"[QA RETRY {attempt + 1}]: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 1.5
                else:
                    raise HTTPException(status_code=500, detail=f"Processing exception: {str(e)}")

# ==================== TASK 2: INVOICE TEXT EXTRACTION ====================
@app.post("/extract")
async def extract_invoice(payload: InvoiceRequest):
    async with RATE_LIMIT_LOCK:
        max_retries = 4
        delay = 1.5
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=payload.invoice_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=InvoiceResponse,
                        system_instruction=(
                            "Extract invoice data into the structured JSON format precisely.\n"
                            "You MUST always populate all 6 fields. Use null if a field cannot be found.\n"
                            "Crucial Date Rule: Convert any human-readable dates (like '15 March 2026') strictly into ISO format 'YYYY-MM-DD'.\n"
                            "Crucial Amount Rule: The 'amount' field MUST be strictly the subtotal BEFORE tax (excluding tax). Do NOT extract the Grand Total or Total After Tax into the 'amount' field.\n"
                            "Crucial Tax Rule: The 'tax' field must be the tax amount only.\n"
                            "Crucial Numeric Rule: Extract numbers as raw floats without commas, currency strings, or extra text symbols.\n"
                            "Crucial Currency Rule: Extract the currency field strictly as a standard 3-letter international ISO currency code (e.g., 'INR', 'USD', 'GBP')."
                        )
                    ),
                )
                
                if not response.text:
                    raise Exception("Empty text string returned during structural extraction")
                
                raw_text = response.text.strip()
                
                # Robust Markdown/Clean parsing to prevent JSON decode errors
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                
                extracted_data = json.loads(raw_text.strip())
                await asyncio.sleep(1.0)
                return extracted_data

            except Exception as e:
                print(f"[EXTRACT RETRY {attempt + 1}]: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                else:
                    raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
