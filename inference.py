
import joblib
import re
from pathlib import Path
from typing import Tuple, List, Dict

MODEL_PATH = Path("models")

# Load models once at module level
try:
    vectorizer = joblib.load(MODEL_PATH / 'vectorizer.pkl')
    model = joblib.load(MODEL_PATH / 'model.pkl')
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load models: {e}")
    print("   Please run 'python train.py' first")
    vectorizer = None
    model = None

# Compiled regex for performance
URL_REGEX = re.compile(r'^https?://(?:www\.)?')

def normalize_url(url: str) -> str:
    """
    Normalize URL by removing protocol and www prefix
    
    Example:
        https://www.example.com → example.com
    """
    return URL_REGEX.sub('', str(url).lower()).rstrip('/')

def predict_url(url: str, threshold: float = 0.42) -> Tuple[bool, float]:
    """
    Predict if a URL is phishing
    
    Args:
        url: URL to check
        threshold: Decision threshold (default 0.42 for high recall)
    
    Returns:
        (is_phishing: bool, probability: float)
    
    Note:
        threshold=0.42 provides ~99.8% recall, ~94% precision
        This catches almost all phishing with few false positives
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Models not loaded. Run train.py first.")
    
    clean = normalize_url(url)
    vec = vectorizer.transform([clean])
    proba = float(model.predict_proba(vec)[0, 1])
    return proba >= threshold, proba

def predict_batch(urls: List[str], threshold: float = 0.42) -> List[Dict]:
    """
    Predict multiple URLs at once
    
    Returns:
        List of dicts with url, is_phishing, probability
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Models not loaded. Run train.py first.")
    
    results = []
    for url in urls:
        try:
            is_phishing, prob = predict_url(url, threshold)
            results.append({
                'url': url,
                'is_phishing': is_phishing,
                'probability': prob,
                'risk_level': get_risk_level(prob)
            })
        except Exception as e:
            results.append({
                'url': url,
                'is_phishing': False,
                'probability': 0.0,
                'risk_level': 'ERROR',
                'error': str(e)
            })
    
    return results

def get_risk_level(probability: float) -> str:
    """Determine risk level based on phishing probability"""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.42:  # Our threshold
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def test_samples():
    """Test with sample URLs"""
    test_urls = [
        "https://www.google.com",
        "https://github.com/user/repo",
        "paypal.com.security-update.verification-2025.net/login",
        "amazon.co.uk.session-id-verify.live",
        "https://www.wikipedia.org",
        "microsoft-account-verify.net/login",
        "bit.ly/3Xa9K2m",
        "secure-bank-login-update-now.ru/account",
        "https://www.reddit.com",
        "gooogle.com/signin"
    ]
    
    print("\n" + "="*90)
    print("PHISHING DETECTION TEST")
    print("="*90)
    print(f"{'URL':<50} {'VERDICT':<10} {'PROBABILITY':<12} {'RISK'}")
    print("-"*90)
    
    for url in test_urls:
        try:
            is_phishing, prob = predict_url(url)
            risk = get_risk_level(prob)
            verdict = "PHISH" if is_phishing else "✅ SAFE"
            
            print(f"{url[:48]:<50} {verdict:<10} {prob:>6.2%}       {risk}")
        except Exception as e:
            print(f"{url[:48]:<50} ERROR   {str(e)}")
    
    print("-"*90)

def interactive_mode():
    """Interactive URL testing"""
    print("\n" + "="*90)
    print("INTERACTIVE PHISHING DETECTOR")
    print("="*90)
    print("Enter URLs to check (or 'quit' to exit)")
    print("Tip: You can enter URLs with or without http://\n")
    
    while True:
        url = input("\nEnter URL: ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("Goo4. Optuna Hyperparameter Optimizationdbye!")
            break
        
        if not url:
            continue
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            is_phishing, prob = predict_url(url)
            risk = get_risk_level(prob)
            
            print("\n" + "─"*70)
            if is_phishing:
                print("PHISHING DETECTED!")
                print(f"   This URL appears to be a phishing attempt")
            else:
                print("LIKELY LEGITIMATE")
                print(f"   This URL appears to be safe")
            
            print(f"\n   Phishing Probability: {prob:.2%}")
            print(f"   Risk Level: {risk}")
            print(f"   Confidence: {max(prob, 1-prob):.2%}")
            
            if is_phishing:
                print("\nWARNING:")
                print("   • Do NOT enter personal information")
                print("   • Do NOT click links or download files")
                print("   • Report this URL if received via email")
            
            print("─"*70)
            
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main function"""
    import sys
    
    if model is None or vectorizer is None:
        print("\nModels not loaded!")
        print("Please run: python train.py")
        return
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--interactive', '-i']:
            interactive_mode()
        elif sys.argv[1] in ['--help', '-h']:
            print("""
Phishing URL Detector - Inference Tool

Usage:
    python inference.py                    # Run test samples
    python inference.py <URL>              # Check single URL
    python inference.py --interactive      # Interactive mode
    python inference.py --help             # Show this help

Examples:
    python inference.py https://example.com
    python inference.py paypal-verify.com/login
    python inference.py -i

Threshold:
    Default threshold = 0.42 (optimized for ~99.8% recall)
    This catches almost all phishing with minimal false positives
            """)
        else:
            # Single URL prediction
            url = sys.argv[1]
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            is_phishing, prob = predict_url(url)
            risk = get_risk_level(prob)
            
            print(f"\n{'='*70}")
            print(f"URL: {url}")
            print(f"{'='*70}")
            print(f"Verdict: {'PHISHING' if is_phishing else '✅ LEGITIMATE'}")
            print(f"Probability: {prob:.4f}")
            print(f"Risk Level: {risk}")
            print(f"{'='*70}\n")
    else:
        test_samples()

if __name__ == "__main__":
    main()