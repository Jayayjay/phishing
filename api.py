"""
Phishing URL Detection - FastAPI with Email Scanning
99.6% AUC - Production-ready API with URL extraction from emails
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from urlextract import URLExtract
from inference import predict_url, predict_batch, get_risk_level
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="Phishing URL Detector API",
    description="ML-powered phishing detection with email scanning (99.6% AUC)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize URL extractor (cached at module level)
url_extractor = URLExtract()

# ==================== Pydantic Models ====================

class URLCheckRequest(BaseModel):
    url: str = Field(..., description="URL to check for phishing")
    threshold: Optional[float] = Field(0.42, ge=0.0, le=1.0, description="Detection threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://paypal-secure-verify.com/login",
                "threshold": 0.42
            }
        }

class URLBatchRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to check", max_length=100)
    threshold: Optional[float] = Field(0.42, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://www.google.com",
                    "http://paypal-verify.suspicious.com"
                ]
            }
        }

class EmailRequest(BaseModel):
    email_text: str = Field(..., description="Email content to scan for phishing URLs")
    threshold: Optional[float] = Field(0.42, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_text": "Click here to verify your account: https://paypal-secure.com/verify"
            }
        }

class PredictionResponse(BaseModel):
    url: str
    is_phishing: bool
    probability: float
    risk_level: str
    confidence: float

class EmailScanResponse(BaseModel):
    status: str
    verdict: str
    phishing_urls: List[str]
    total_urls: int
    phishing_count: int
    details: List[Dict]

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool

# ==================== Endpoints ====================

@app.get("/", response_model=Dict)
async def root():
    """API information and available endpoints"""
    return {
        "name": "Phishing URL Detector API",
        "version": "2.0.0",
        "model_accuracy": "99.6% AUC",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch": "POST /predict/batch",
            "email_scan": "POST /scan",
            "quick_check": "GET /check?url=<url>",
            "documentation": "GET /docs"
        },
        "features": [
            "Single URL checking",
            "Batch URL processing",
            "Email content scanning",
            "Automatic URL extraction",
            "Risk level assessment"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        from inference import model, vectorizer
        model_loaded = model is not None and vectorizer is not None
    except:
        model_loaded = False
    
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "version": "2.0.0",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: URLCheckRequest):
    """
    Predict if a single URL is phishing
    
    Returns detailed analysis with probability and risk level
    """
    try:
        is_phishing, prob = predict_url(request.url, threshold=request.threshold)
        
        return PredictionResponse(
            url=request.url,
            is_phishing=is_phishing,
            probability=prob,
            risk_level=get_risk_level(prob),
            confidence=max(prob, 1 - prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_urls_batch(request: URLBatchRequest):
    """
    Predict multiple URLs at once (max 100)
    
    Efficiently processes batch requests
    """
    if len(request.urls) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 URLs per batch")
    
    try:
        results = predict_batch(request.urls, threshold=request.threshold)
        
        return [
            PredictionResponse(
                url=r['url'],
                is_phishing=r['is_phishing'],
                probability=r['probability'],
                risk_level=r['risk_level'],
                confidence=max(r['probability'], 1 - r['probability'])
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/scan", response_model=EmailScanResponse)
async def scan_email(payload: EmailRequest):
    """
    MAIN FEATURE: Scan email content for phishing URLs
    
    Automatically extracts all URLs from email text and checks each one.
    Perfect for email security gateways and user-facing tools.
    
    Example use cases:
    - Email security scanning
    - Phishing detection in messages
    - Automated threat analysis
    """
    try:
        # Extract all URLs from email text
        urls = url_extractor.find_urls(payload.email_text)
        
        if not urls:
            return EmailScanResponse(
                status="safe",
                verdict="No URLs found in email",
                phishing_urls=[],
                total_urls=0,
                phishing_count=0,
                details=[]
            )
        
        # Predict each URL
        results = [predict_url(url, threshold=payload.threshold) for url in urls]
        
        # Identify phishing URLs
        phishing = [url for url, (bad, _) in zip(urls, results) if bad]
        
        # Create detailed results
        details = []
        for url, (is_phishing, prob) in zip(urls, results):
            details.append({
                "url": url,
                "is_phishing": is_phishing,
                "score": float(prob),
                "risk_level": get_risk_level(prob)
            })
        
        # Determine overall verdict
        if phishing:
            status = "phishing"
            verdict = f"PHISHING DETECTED - {len(phishing)} malicious URL(s) found"
        else:
            status = "safe"
            verdict = f"All {len(urls)} URL(s) appear safe"
        
        return EmailScanResponse(
            status=status,
            verdict=verdict,
            phishing_urls=phishing,
            total_urls=len(urls),
            phishing_count=len(phishing),
            details=details
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email scan error: {str(e)}")

@app.get("/check", response_model=PredictionResponse)
async def quick_check(url: str, threshold: float = 0.42):
    """
    Quick URL check via GET request
    
    Example: /check?url=https://example.com&threshold=0.5
    """
    try:
        is_phishing, prob = predict_url(url, threshold=threshold)
        
        return PredictionResponse(
            url=url,
            is_phishing=is_phishing,
            probability=prob,
            risk_level=get_risk_level(prob),
            confidence=max(prob, 1 - prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get model statistics and information
    """
    try:
        from inference import model, vectorizer
        
        if model is None or vectorizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "model": {
                "type": "LightGBM",
                "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
                "feature_count": len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else None
            },
            "performance": {
                "auc": 0.996,
                "recall": 0.998,
                "precision": 0.94,
                "default_threshold": 0.42
            },
            "features": {
                "type": "Character n-grams",
                "ngram_range": "(2, 6)",
                "max_features": 1000000,
                "vectorization": "TF-IDF"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

if __name__ == "__main__":
    print("Starting Phishing Detector API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)