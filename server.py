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
                
                return {"answer": response.text.strip()}

            except Exception as e:
                print(f"[QA RETRY {attempt + 1}]: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = 15.0 if "RESOURCE_EXHAUSTED" in str(e) else 2.0
                    await asyncio.sleep(sleep_time)
                else:
                    raise HTTPException(status_code=500, detail=f"Processing exception: {str(e)}")

# ==================== TASK 2: INVOICE TEXT EXTRACTION ====================
@app.post("/extract")
async def extract_invoice(payload: InvoiceRequest):
    async with RATE_LIMIT_LOCK:
        max_retries = 4
        for attempt in range(max_retries):
            try:
                structured_prompt = (
                    f"Text payload:\n{payload.invoice_text}\n\n"
                    "Task: Extract corporate invoice metrics strictly matching the properties key details listed down below.\n"
                    "Your response format MUST be a pure valid minified JSON dictionary string containing exactly these 6 keys. If a value cannot be found, populate it as null.\n\n"
                    "Expected Keys JSON Structure Guideline:\n"
                    "{\n"
                    "  \"invoice_no\": \"String token value or null\",\n"
                    "  \"date\": \"Strict ISO string format 'YYYY-MM-DD' only, or null\",\n"
                    "  \"vendor\": \"String organization name or null\",\n"
                    "  \"amount\": Raw subtotal float value BEFORE tax only, or null,\n"
                    "  \"tax\": Raw float tax amount value only, or null,\n"
                    "  \"currency\": \"Strict 3-letter international ISO currency code string (e.g., 'INR', 'USD', 'GBP'), or null\"\n"
                    "}\n\n"
                    "Important Parsing Rule: For 'amount', use the raw subtotal before tax. Clean all values from local signs, extra letters, or commas."
                )

                # Strict application/json mime type configuration setup without Pydantic schema bug
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=structured_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                if not response.text:
                    raise Exception("Blank textual payload response detected from Gemini")
                
                raw_text = response.text.strip()
                
                if "```" in raw_text:
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                
                extracted_data = json.loads(raw_text.strip())
                return extracted_data

            except Exception as e:
                print(f"[EXTRACT RETRY {attempt + 1}]: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = 15.0 if "RESOURCE_EXHAUSTED" in str(e) else 2.0
                    await asyncio.sleep(sleep_time)
                else:
                    raise HTTPException(status_code=500, detail=f"Extraction failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
