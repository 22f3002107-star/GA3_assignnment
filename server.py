import base64
import io
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from PIL import Image

app = FastAPI()

# Enable CORS for the grader cloudflare workers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Client setup to reuse connections
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")
client = genai.Client(api_key=API_KEY)

# Strict Lock for Rate Limiting (Ensures orderly processing)
RATE_LIMIT_LOCK = asyncio.Lock()

class QARequest(BaseModel):
    image_base64: str
    question: str

def decode_image_helper(base64_str: str) -> Image.Image:
    """Safely decodes base64 string to PIL Image."""
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
        
        for attempt in range(max_retries):
            try:
                # Decode the incoming image
                image = decode_image_helper(payload.image_base64)
                
                # Dynamic global instruction following Rule 1 strictly
                prompt = (
                    f"Question: {payload.question}\n\n"
                    "Task: Answer the question directly based on the provided image (it could be a document, invoice, chart, or table).\n"
                    "Strict Rule for numbers: If the answer is a numeric value (like a total amount, tax, or chart data point), output ONLY the raw number digits (e.g., 4089.35 or 120). Do not include any words, commas, letters, currency symbols, or units.\n"
                    "Strict Rule for text: If the answer is text (like a name or date), output just the direct answer plainly without extra conversational text."
                )
                
                # Standard model call for all types of document images
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image]
                )
                
                if not response.text:
                    raise Exception("Empty response from Gemini API")
                
                # Strict 4.5s delay to safely respect Gemini free tier limits
                await asyncio.sleep(4.5)
                
                return {"answer": response.text.strip()}

            except Exception as e:
                print(f"[QA RETRY {attempt + 1}]: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise HTTPException(status_code=500, detail=f"Processing exception: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
