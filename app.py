"""
Phishing URL Detection - Streamlit Web Interface
Advanced demo with URL checking and email scanning
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from urlextract import URLExtract
from inference import predict_url, predict_batch, get_risk_level
import time

# Page config
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .phishing-alert {
        padding: 20px;
        background-color: #ff4444;
        color: white;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .safe-alert {
        padding: 20px;
        background-color: #00C851;
        color: white;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .warning-alert {
        padding: 20px;
        background-color: #ffbb33;
        color: #333;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_extractor():
    """Load URL extractor (cached)"""
    return URLExtract()

def create_gauge_chart(probability):
    """Create a gauge chart for phishing probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Phishing Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#00ff00'},
                {'range': [20, 40], 'color': '#00cc00'},
                {'range': [40, 60], 'color': '#ffcc00'},
                {'range': [60, 80], 'color': '#ff6600'},
                {'range': [80, 100], 'color': '#ff0000'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 42  # Our optimal threshold
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_risk_distribution(results):
    """Create pie chart of risk levels"""
    risk_counts = {}
    for r in results:
        risk = r.get('risk_level', 'UNKNOWN')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    fig = px.pie(
        values=list(risk_counts.values()),
        names=list(risk_counts.keys()),
        title="Risk Level Distribution",
        color_discrete_map={
            'MINIMAL': '#00ff00',
            'LOW': '#00cc00',
            'MEDIUM': '#ffcc00',
            'HIGH': '#ff6600',
            'CRITICAL': '#ff0000'
        }
    )
    fig.update_traces(textinfo='label+percent')
    return fig

def main():
    # Header
    st.title("Phishing URL Detector")
    st.markdown("### Advanced AI-powered protection against phishing attacks")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool uses LightGBM with character n-gram analysis to detect phishing URLs.
        
        **Model Performance:**
        - 99.6% AUC
        - 99.8% Recall at 0.42 threshold
        - 94% Precision
        
        **Detection Methods:**
        - Character pattern analysis
        - TF-IDF vectorization
        - Gradient boosting classification
        """)
        
        st.header("Settings")
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.42,
            step=0.01,
            help="Lower = more sensitive (catch more phishing, more false positives)"
        )
        
        st.header("Example URLs")
        st.code("https://www.google.com", language="text")
        st.code("paypal-verify.suspicious.com", language="text")
        st.code("amazon.co.uk.verify-account.live", language="text")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single URL Check", 
        "Batch Check", 
        "Email Scanner", 
        "Statistics"
    ])
    
    # ==================== TAB 1: Single URL ====================
    with tab1:
        st.header("Check a Single URL")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url = st.text_input(
                "Enter URL to check:",
                placeholder="https://example.com",
                help="Enter the full URL including http:// or https://"
            )
        
        with col2:
            st.write("")
            st.write("")
            check_button = st.button("Analyze URL", type="primary", use_container_width=True)
        
        if check_button and url:
            with st.spinner("Analyzing URL..."):
                time.sleep(0.3)
                
                try:
                    is_phishing, prob = predict_url(url, threshold=threshold)
                    risk = get_risk_level(prob)
                    
                    st.markdown("---")
                    
                    # Display result
                    if is_phishing:
                        st.markdown(
                            '<div class="phishing-alert">PHISHING DETECTED</div>',
                            unsafe_allow_html=True
                        )
                        st.error("This URL appears to be a phishing attempt. Do NOT enter personal information!")
                    else:
                        st.markdown(
                            '<div class="safe-alert">LIKELY LEGITIMATE</div>',
                            unsafe_allow_html=True
                        )
                        st.success("This URL appears to be safe, but always exercise caution.")
                    
                    st.markdown("---")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Prediction",
                            "PHISHING" if is_phishing else "LEGITIMATE",
                            delta="High Risk" if is_phishing else "Low Risk",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{max(prob, 1-prob):.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Risk Level",
                            risk
                        )
                    
                    # Gauge chart
                    st.plotly_chart(
                        create_gauge_chart(prob),
                        use_container_width=True
                    )
                    
                    # Detailed probabilities
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Phishing Probability:**")
                        st.progress(prob)
                        st.write(f"{prob:.2%}")
                    
                    with col2:
                        st.write("**Legitimate Probability:**")
                        st.progress(1 - prob)
                        st.write(f"{1-prob:.2%}")
                
                except Exception as e:
                    st.error(f"Error analyzing URL: {e}")
        
        elif check_button:
            st.warning("Please enter a URL to check")
    
    # ==================== TAB 2: Batch Check ====================
    with tab2:
        st.header("Batch URL Checker")
        st.write("Check multiple URLs at once (one per line)")
        
        urls_text = st.text_area(
            "Enter URLs:",
            height=200,
            placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com"
        )
        
        if st.button("Analyze All URLs", type="primary"):
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            if urls:
                progress_bar = st.progress(0)
                results = []
                
                for i, url in enumerate(urls):
                    try:
                        is_phishing, prob = predict_url(url, threshold=threshold)
                        results.append({
                            'url': url,
                            'is_phishing': is_phishing,
                            'probability': prob,
                            'risk_level': get_risk_level(prob)
                        })
                    except Exception as e:
                        st.warning(f"Error processing {url}: {e}")
                    
                    progress_bar.progress((i + 1) / len(urls))
                
                st.success(f"Analyzed {len(results)} URLs")
                
                # Summary metrics
                phishing_count = sum(1 for r in results if r['is_phishing'])
                legitimate_count = len(results) - phishing_count
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Checked", len(results))
                
                with col2:
                    st.metric(
                        "Phishing Detected", 
                        phishing_count, 
                        delta=f"{phishing_count/len(results):.1%}"
                    )
                
                with col3:
                    st.metric(
                        "Legitimate", 
                        legitimate_count, 
                        delta=f"{legitimate_count/len(results):.1%}"
                    )
                
                # Risk distribution chart
                st.plotly_chart(create_risk_distribution(results), use_container_width=True)
                
                # Results table
                st.markdown("---")
                st.subheader("Results")
                
                df_results = pd.DataFrame([{
                    'URL': r['url'],
                    'Status': 'PHISHING' if r['is_phishing'] else 'LEGITIMATE',
                    'Phishing Probability': f"{r['probability']:.2%}",
                    'Risk Level': r['risk_level']
                } for r in results])
                
                # Color code the status
                def highlight_status(row):
                    if row['Status'] == 'PHISHING':
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return ['background-color: #ccffcc'] * len(row)
                
                st.dataframe(
                    df_results.style.apply(highlight_status, axis=1),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Please enter at least one URL")
    
    # ==================== TAB 3: Email Scanner ====================
    with tab3:
        st.header("Email Content Scanner")
        st.write("Paste email content to automatically extract and check all URLs")
        
        email_text = st.text_area(
            "Email Content:",
            height=300,
            placeholder="""Paste your email here...

Example:
Dear Customer,

Your account requires verification. Please visit:
https://paypal-secure-login.com/verify

Thank you,
PayPal Security Team
            """,
            help="The tool will automatically find and check all URLs in the text"
        )
        
        if st.button("Scan Email", type="primary"):
            if email_text.strip():
                with st.spinner("Extracting and analyzing URLs..."):
                    try:
                        # Extract URLs
                        extractor = load_extractor()
                        urls = extractor.find_urls(email_text)
                        
                        if not urls:
                            st.info("No URLs found in the email content")
                        else:
                            st.success(f"Found {len(urls)} URL(s) in email")
                            
                            # Analyze each URL
                            results = []
                            for url in urls:
                                try:
                                    is_phishing, prob = predict_url(url, threshold=threshold)
                                    results.append({
                                        'url': url,
                                        'is_phishing': is_phishing,
                                        'probability': prob,
                                        'risk_level': get_risk_level(prob)
                                    })
                                except Exception as e:
                                    st.warning(f"Error checking {url}: {e}")
                            
                            # Overall verdict
                            phishing_urls = [r['url'] for r in results if r['is_phishing']]
                            
                            st.markdown("---")
                            
                            if phishing_urls:
                                st.markdown(
                                    '<div class="phishing-alert">PHISHING EMAIL DETECTED</div>',
                                    unsafe_allow_html=True
                                )
                                st.error(f"""
                                **DANGER: {len(phishing_urls)} malicious URL(s) detected!**
                                
                                This email appears to be a phishing attempt.
                                
                                **DO NOT:**
                                - Click any links
                                - Enter personal information
                                - Download attachments
                                - Reply to this email
                                
                                **Recommended Actions:**
                                - Delete this email immediately
                                - Report as phishing/spam
                                - Check your account through official channels only
                                """)
                            else:
                                st.markdown(
                                    '<div class="safe-alert">EMAIL APPEARS SAFE</div>',
                                    unsafe_allow_html=True
                                )
                                st.success(f"All {len(urls)} URL(s) in this email appear legitimate.")
                            
                            st.markdown("---")
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("URLs Found", len(urls))
                            
                            with col2:
                                st.metric(
                                    "Phishing URLs", 
                                    len(phishing_urls),
                                    delta="DANGER" if phishing_urls else "SAFE",
                                    delta_color="inverse"
                                )
                            
                            with col3:
                                st.metric("Safe URLs", len(urls) - len(phishing_urls))
                            
                            # Detailed results
                            st.subheader("URL Analysis Details")
                            
                            for r in results:
                                with st.expander(
                                    f"{'[PHISHING]' if r['is_phishing'] else '[SAFE]'} {r['url'][:60]}..."
                                ):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Status", "PHISHING" if r['is_phishing'] else "LEGITIMATE")
                                    
                                    with col2:
                                        st.metric("Probability", f"{r['probability']:.2%}")
                                    
                                    with col3:
                                        st.metric("Risk Level", r['risk_level'])
                                    
                                    st.write(f"**Full URL:** `{r['url']}`")
                    
                    except Exception as e:
                        st.error(f"Error scanning email: {e}")
            else:
                st.warning("Please enter email content to scan")
    
    # ==================== TAB 4: Statistics ====================
    with tab4:
        st.header("Model Statistics & Information")
        
        st.info("""
        This phishing detector uses state-of-the-art machine learning techniques
        to identify malicious URLs with high accuracy.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.write("""
            **Algorithm:** LightGBM (Gradient Boosting)
            
            **Features:** Character n-grams (2-6 chars)
            
            **Vectorization:** TF-IDF (1M features)
            
            **Training:** Optuna hyperparameter optimization
            
            **Threshold:** 0.42 (optimized for recall)
            """)
            
            st.subheader("Performance Metrics")
            st.write("""
            **ROC-AUC:** 99.6%
            
            **Recall:** 99.8% (catches nearly all phishing)
            
            **Precision:** 94% (low false positive rate)
            
            **Accuracy:** ~97%
            
            **Training Data:** Kaggle Phishing Site URLs Dataset
            """)
        
        with col2:
            st.subheader("How It Works")
            st.write("""
            **1. URL Normalization**
            - Removes http/https and www
            - Converts to lowercase
            
            **2. Character N-Gram Extraction**
            - Breaks URL into overlapping sequences
            - Example: "paypal" → "pa", "ay", "yp", "pa", "al"
            
            **3. TF-IDF Vectorization**
            - Weights patterns by importance
            - Creates 1M dimensional feature vector
            
            **4. Gradient Boosting Prediction**
            - 10,000 decision trees
            - Ensemble voting
            - Probability output
            
            **5. Risk Assessment**
            - Applies optimal threshold
            - Categorizes risk level
            - Returns verdict
            """)
            
            st.subheader("Why Character N-Grams?")
            st.write("""
            Character n-grams are excellent for phishing detection because they:
            
            - Detect typosquatting (gooogle.com vs google.com)
            - Catch suspicious patterns (verify-account-login)
            - Work in any language
            - Require no manual feature engineering
            - Capture subtle URL structures
            - Adapt to new phishing techniques
            """)
        
        st.markdown("---")
        
        st.subheader("Risk Level Definitions")
        
        risk_table = pd.DataFrame({
            'Risk Level': ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            'Probability Range': ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
            'Description': [
                'Very unlikely to be phishing',
                'Probably legitimate',
                'Suspicious, exercise caution',
                'Likely phishing attempt',
                'Almost certainly phishing'
            ]
        })
        
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()