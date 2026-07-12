import base64
import io
import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types  
from PIL import Image
from typing import Optional, List

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

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None  
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None  

class RowItem(BaseModel):
    label: str               
    value: float             

class TableExtractionResponse(BaseModel):
    title: Optional[str] = None
    data_points: List[RowItem]

def decode_image_helper(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[-1]
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))

@app.post("/answer-image")
async def answer_image(payload: QARequest):
    async with RATE_LIMIT_LOCK:
        max_retries = 4
        delay = 2
        
        question_lower = payload.question.lower()
        is_invoice_task = "invoice" in question_lower or "bill" in question_lower or "vendor" in question_lower
        is_table_task = "table" in question_lower or "chart" in question_lower or "pie" in question_lower or "bar" in question_lower or "data point" in question_lower

        for attempt in range(max_retries):
            try:
                image = decode_image_helper(payload.image_base64)
                
                # ---------------- TASK 2: INVOICE DATA EXTRACTION ----------------
                if is_invoice_task:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[payload.question, image],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InvoiceResponse,
                            system_instruction=(
                                "Extract invoice data into structured JSON format perfectly.\n"
                                "Crucial Date Rule: Convert any human-readable dates strictly into ISO format 'YYYY-MM-DD'.\n"
                                "Crucial Numeric Rule: Extract numbers as raw floats without commas or currency characters.\n"
                                "Crucial Currency Rule: Convert currency strings or symbols strictly to their 3-letter international ISO code (e.g., 'INR', 'USD', 'GBP')."
                            )
                        ),
                    )
                    if not response.text:
                        raise Exception("Empty response text in Invoice task")
                    
                    await asyncio.sleep(4.5)
                    # Force response format to strict single string dictionary wrapper object
                    return {"answer": json.dumps(json.loads(response.text.strip()))}

                # ---------------- TASK 3: TABULAR / CHART EXTRACTION ----------------
                elif is_table_task:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[payload.question, image],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=TableExtractionResponse,
                            system_instruction=(
                                "Extract data from tables, bar charts, or pie charts into structured data point objects.\n"
                                "Identify the main title of the chart/table and mapping elements directly.\n"
                                "Ensure numbers are parsed clean as standard float/int representations without external noise labels."
                            )
                        ),
                    )
                    if not response.text:
                        raise Exception("Empty response text in Table task")
                    
                    await asyncio.sleep(4.5)
                    # Force response format to strict single string dictionary wrapper object
                    return {"answer": json.dumps(json.loads(response.text.strip()))}

                # ---------------- TASK 1: DIRECT RAW QA PROMPT ----------------
                else:
                    prompt = (
                        f"Question: {payload.question}\n\n"
                        "Task: Answer the question directly based on the image.\n"
                        "Strict Rule for numbers: If the answer is a numeric value, output ONLY the raw number digits (e.g., 4089.35). Do not include any words, commas, letters, currency symbols, or units."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt, image]
                    )
                    if not response.text:
                        raise Exception("Empty response text in direct QA task")
                        
                    await asyncio.sleep(4.5)
                    return {"answer": response.text.strip()}

            except Exception as e:
                print(f"[API ROUTER QUEUE RETRY {attempt + 1}]: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise HTTPException(status_code=500, detail=f"Critical processing exception: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
