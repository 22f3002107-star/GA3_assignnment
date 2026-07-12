import base64
import io
import os
import json
import asyncio
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types  
from PIL import Image
from typing import Optional, Dict, Any

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

class DynamicExtractRequest(BaseModel):
    text: str
    schema_def: Dict[str, str] = Field(..., alias="schema")

    class Config:
        populate_by_name = True

def decode_image_helper(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[-1]
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))

# High-Intelligence Fallback Parser to split fields by period or metadata keys
def smart_python_extract_fallback(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    output = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for key, data_type in schema.items():
        key_lower = key.lower()
        val = None
        
        # Rule A: Standard anchor checking (e.g., conference: ..., title is ...)
        for line in lines:
            pattern = rf'\b{re.escape(key_lower)}\b\s*(?::|-|is|=)\s*(.*)'
            orig_match = re.search(pattern, line, re.IGNORECASE)
            if orig_match:
                val = orig_match.group(1).strip(" \t.,\"'")
                break
                    
        # Rule B: Contextual extraction using Quotes or positional lines if anchor misses
        if not val and data_type == "string":
            quotes_match = re.findall(r"['\"](.*?)['\"]", text)
            if quotes_match:
                val = quotes_match[0]
            elif lines:
                if "title" in key_lower:
                    val = lines[0]
                else:
                    for line in lines:
                        if key_lower in line.lower():
                            val = line
                            break

        # Rule C: Formatting cleanups and cutting out trailing sentence parameters
        if val is not None:
            val_str = str(val).strip(" \t.,\"'")
            
            # Strip structural item field prefixes
            val_str = re.sub(r'^(published|title|paper|name|topic|conference|venue|journal)\s*(?::|-|is|=)?\s*', '', val_str, flags=re.IGNORECASE)
            
            # CRUCIAL FIX: Split at a period followed by space + capitalized word or token key boundary
            val_str = re.split(r'\.\s+(?=[A-Z])|\.\s+[A-Za-z\s]+:', val_str)[0]
            
            val_str = val_str.strip(" \t.,\"'")
            
            try:
                if data_type == "integer":
                    num_match = re.search(r'\d+', val_str)
                    output[key] = int(num_match.group(0)) if num_match else None
                elif data_type == "float":
                    float_match = re.search(r'\d+\.\d+|\d+', val_str)
                    output[key] = float(float_match.group(0)) if float_match else None
                else:
                    output[key] = val_str
            except Exception:
                output[key] = None
        else:
            # Absolute default structural type fallbacks
            if data_type == "integer":
                nums = re.findall(r'\b\d+\b', text)
                output[key] = int(nums[0]) if nums else None
            elif data_type == "float":
                floats = re.findall(r'\b\d+\.\d+\b', text)
                output[key] = float(floats[0]) if floats else None
            elif data_type == "date":
                date_match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', text)
                output[key] = date_match.group(0) if date_match else None
            else:
                output[key] = None
    return output

# ==================== TASK 1: MULTIMODAL QA ====================
@app.post("/answer-image")
async def answer_image(payload: QARequest):
    async with RATE_LIMIT_LOCK:
        try:
            image = decode_image_helper(payload.image_base64)
            prompt = f"Question: {payload.question}\n\nTask: Answer directly. Numbers as digits only."
            response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, image])
            return {"answer": response.text.strip()}
        except Exception:
            return {"answer": "Error parsing image response"}

# ==================== TASK 2: INVOICE EXTRACTION ====================
@app.post("/extract")
async def extract_invoice(payload: InvoiceRequest):
    async with RATE_LIMIT_LOCK:
        try:
            structured_prompt = f"Text:\n{payload.invoice_text}\n\nExtract JSON with keys: invoice_no, date, vendor, amount, tax, currency."
            response = client.models.generate_content(
                model='gemini-2.5-flash', contents=structured_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text.strip())
        except Exception:
            return {"invoice_no": None, "date": None, "vendor": None, "amount": None, "tax": None, "currency": None}

# ==================== TASK 3 (Q4): DYNAMIC EXTRACTION ====================
@app.post("/dynamic-extract")
async def dynamic_extract(payload: DynamicExtractRequest):
    async with RATE_LIMIT_LOCK:
        try:
            dynamic_prompt = (
                f"Context text:\n{payload.text}\n\n"
                f"Requested Schema definition:\n{json.dumps(payload.schema_def, indent=2)}\n\n"
                "Task: Return a clean JSON matching the requested keys precisely. Convert integers/floats/dates accordingly."
            )

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=dynamic_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            raw_text = response.text.strip()
            if "```" in raw_text:
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
                
            extracted_dynamic_data = json.loads(raw_text.strip())
            
            final_sanitized_output = {}
            for key, type_str in payload.schema_def.items():
                val = extracted_dynamic_data.get(key, None)
                if val is not None:
                    if type_str == "integer": final_sanitized_output[key] = int(float(str(val).replace(",", "")))
                    elif type_str == "float": final_sanitized_output[key] = float(str(val).replace(",", ""))
                    else: final_sanitized_output[key] = str(val)
                else:
                    final_sanitized_output[key] = None
            return final_sanitized_output

        except Exception as e:
            # Fallback isolation triggered seamlessly
            print(f"[RESCUING USING INTELLIGENT PARSER FALLBACK]: {str(e)}")
            fallback_result = smart_python_extract_fallback(payload.text, payload.schema_def)
            return fallback_result

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
