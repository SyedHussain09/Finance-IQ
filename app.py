"""
💎 FinanceIQ - AI-Powered Financial Analytics Platform
======================================================
Developed by: Syed Sajjad Hussain
Year: 2026
Version: 1.0.0

Advanced Financial Intelligence with Anthropic 
Smooth, Accurate, and Lightning-Fast Analytics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables - use explicit path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)
from scipy import stats
import time
from io import BytesIO
import base64

# PDF Generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import AI modules
try:
    from smart_ai_engine import UltraSmartAI
    from smart_visualizations import SmartVisualizer
    SMART_MODULES = True
except:
    SMART_MODULES = False

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="FinanceIQ - AI Financial Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
session_defaults = {
    'chat_history': [],
    'analysis_done': False,
    'analytics_data': None,
    'df': None,
    'response_mode': 'detailed',
    'show_welcome': True,
    'page': '📤 Upload Data',
    'pdf_ready': False,
    'pdf_buffer': None
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== API CONFIGURATION ====================
def get_api_key(name):
    """Securely retrieve API key from secrets or environment"""
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except:
        return os.getenv(name, "")

API_KEY = get_api_key("ANTHROPIC_API_KEY")

# Validate API Key
if API_KEY and API_KEY.startswith("sk-ant-"):
    print(f"\n{'='*60}")
    print(f"✅ ANTHROPIC API KEY LOADED SUCCESSFULLY")
    print(f"   Key preview: {API_KEY[:15]}...{API_KEY[-10:]}")
    print(f"   Key length: {len(API_KEY)} characters")
    print(f"{'='*60}\n")
else:
    print(f"\n{'='*60}")
    print(f"⚠️  WARNING: ANTHROPIC API KEY NOT CONFIGURED PROPERLY")
    print(f"   Current value: {API_KEY[:20] if API_KEY else 'None'}...")
    print(f"   Please set your API key in the .env file")
    print(f"{'='*60}\n")

# Initialize AI Engine
ultra_ai = None
smart_viz = None

if SMART_MODULES:
    try:
        print(f"\n🚀 Initializing AI Engine...")
        ultra_ai = UltraSmartAI(anthropic_key=API_KEY if API_KEY else None)
        smart_viz = SmartVisualizer()
        
        if ultra_ai and ultra_ai.anthropic_client:
            print(f"✅ AI Engine initialized successfully")
            print(f"   Model: {ultra_ai.claude_model}")
            print(f"   Status: READY FOR REAL-TIME RESPONSES\n")
        else:
            print(f"⚠️  AI Engine initialized in FALLBACK MODE")
            print(f"   Claude API not available - using rule-based responses\n")
            
    except Exception as e:
        print(f"\n❌ AI initialization error: {e}\n")
        st.error(f"AI initialization warning: {e}")

# ==================== DATABASE ====================
DB_PATH = Path("financeiq.db")

def init_database():
    """Initialize SQLite database with optimized schema"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_database()

def load_from_database():
    """Load existing data from database if available"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM expenses", conn)
        conn.close()
        
        if len(df) > 0:
            # Convert date column
            df['Date'] = pd.to_datetime(df['date'])
            df['Category'] = df['category']
            df['Amount'] = pd.to_numeric(df['amount'])
            df = df[['Date', 'Category', 'Amount']].copy()
            df = df.dropna()
            df = df.sort_values('Date')
            
            # Update session state
            st.session_state.df = df
            
            # Run analytics automatically
            analytics = perform_advanced_analytics(df)
            st.session_state.analytics_data = analytics
            st.session_state.analysis_done = True
            
            return True
    except Exception as e:
        pass
    return False

# Load existing data on startup
if st.session_state.df is None:
    load_from_database()

# ==================== ANALYTICS ENGINE ====================
def perform_advanced_analytics(df):
    """
    Ultra-Smart Financial Analytics Engine
    Handles all data types with precision and accuracy
    """
    # Deep copy to avoid mutations
    df_clean = df.copy()
    
    # Robust data cleaning and validation
    df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Amount'])
    
    if len(df_clean) == 0:
        return get_empty_analytics()
    
    # Separate positive and negative amounts (expenses vs income/refunds)
    df_expenses = df_clean[df_clean['Amount'] > 0].copy()  # Changed >= to > to exclude zeros
    df_income = df_clean[df_clean['Amount'] <= 0].copy()
    
    # Work with actual expense amounts (already positive)
    expense_amounts = df_expenses['Amount'].values if len(df_expenses) > 0 else np.array([0])
    
    # Core statistics with robust handling
    total = float(df_expenses['Amount'].sum()) if len(df_expenses) > 0 else 0.0
    mean = float(df_expenses['Amount'].mean()) if len(df_expenses) > 0 else 0.0
    median = float(df_expenses['Amount'].median()) if len(df_expenses) > 0 else 0.0
    std = float(df_expenses['Amount'].std()) if len(df_expenses) > 1 else 0.0
    
    # Category intelligence with accurate aggregation
    cat_breakdown = {}
    if len(df_expenses) > 0:
        cat_breakdown = df_expenses.groupby('Category')['Amount'].sum().to_dict()
        # Round to 2 decimals for accuracy
        cat_breakdown = {k: round(float(v), 2) for k, v in cat_breakdown.items()}
    
    cat_stats = {}
    if len(df_expenses) > 0:
        cat_stats = df_expenses.groupby('Category').agg({
            'Amount': ['sum', 'mean', 'count', 'min', 'max']
        }).round(2).to_dict()
    
    # Smart anomaly detection using IQR method
    anomalies = []
    if len(expense_amounts) > 4:  # Need at least 5 data points
        q1, q3 = np.percentile(expense_amounts, [25, 75])
        iqr = q3 - q1
        if iqr > 0:  # Avoid division issues
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = df_expenses[
                (df_expenses['Amount'] < lower_bound) | 
                (df_expenses['Amount'] > upper_bound)
            ].index.tolist()
    
    # Advanced trend analysis with robust regression
    slope = 0
    r_squared = 0
    if len(expense_amounts) > 2:
        x = np.arange(len(expense_amounts))
        try:
            # Use polyfit for better numerical stability
            coeffs = np.polyfit(x, expense_amounts, 1)
            slope = coeffs[0]
            
            # Calculate R-squared for trend reliability
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((expense_amounts - y_pred) ** 2)
            ss_tot = np.sum((expense_amounts - np.mean(expense_amounts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except:
            slope = 0
            r_squared = 0
    
    # Intelligent forecasting with multiple methods
    # Method 1: Trend-based (linear regression)
    trend_7 = mean + slope * 7
    trend_30 = mean + slope * 30
    trend_90 = mean + slope * 90
    
    # Method 2: Moving average based
    ma_window = min(7, len(expense_amounts))
    recent_ma = float(np.mean(expense_amounts[-ma_window:])) if len(expense_amounts) >= ma_window else mean
    
    # Method 3: Weighted combination (more recent = more weight)
    if len(expense_amounts) >= 3:
        weights = np.exp(np.linspace(-1, 0, len(expense_amounts)))
        weights = weights / weights.sum()
        weighted_avg = float(np.average(expense_amounts, weights=weights))
    else:
        weighted_avg = mean
    
    # Smart forecast: combine methods with reliability weighting
    reliability = min(1.0, r_squared + 0.3)  # Boost reliability slightly
    
    forecast_7 = max(0, trend_7 * reliability + weighted_avg * (1 - reliability))
    forecast_30 = max(0, trend_30 * reliability + weighted_avg * (1 - reliability))
    forecast_90 = max(0, trend_90 * reliability + weighted_avg * (1 - reliability))
    
    # Advanced financial health score (0-100)
    
    # 1. Consistency score (lower variance = better)
    cv = (std / mean) if mean > 0 else 1  # Coefficient of variation
    consistency = max(0, min(100, 100 - (cv * 100)))
    
    # 2. Diversity score (more categories = better budgeting)
    num_cats = df_clean['Category'].nunique()
    diversity = min(100, num_cats * 15)  # Each category worth 15 points
    
    # 3. Anomaly impact (fewer anomalies = better)
    anomaly_rate = len(anomalies) / len(df_expenses) if len(df_expenses) > 0 else 0
    anomaly_impact = max(0, 100 - (anomaly_rate * 200))
    
    # 4. Trend stability (stable or decreasing spending = better)
    if abs(slope) < mean * 0.01:  # Nearly flat
        trend_stability = 100
    elif slope < 0:  # Decreasing spending
        trend_stability = 90
    else:  # Increasing spending
        trend_stability = max(0, 100 - abs(slope / mean * 100)) if mean > 0 else 50
    
    # 5. Budget adherence (consistency over time)
    if len(expense_amounts) >= 7:
        weekly_variance = np.std([
            np.mean(expense_amounts[i:i+7]) 
            for i in range(0, len(expense_amounts)-6, 7)
        ])
        budget_adherence = max(0, min(100, 100 - (weekly_variance / mean * 100))) if mean > 0 else 50
    else:
        budget_adherence = consistency
    
    # Weighted health score
    health_score = (
        consistency * 0.30 + 
        diversity * 0.20 + 
        anomaly_impact * 0.20 + 
        trend_stability * 0.20 +
        budget_adherence * 0.10
    )
    
    # Spending velocity (rate of change)
    velocity = 0
    if len(expense_amounts) >= 14:
        recent_period = np.mean(expense_amounts[-7:])
        previous_period = np.mean(expense_amounts[-14:-7])
        velocity = ((recent_period - previous_period) / previous_period * 100) if previous_period > 0 else 0
    elif len(expense_amounts) >= 6:
        recent = np.mean(expense_amounts[-3:])
        previous = np.mean(expense_amounts[:3])
        velocity = ((recent - previous) / previous * 100) if previous > 0 else 0
    
    # Top spending categories (for quick insights)
    top_categories = sorted(
        cat_breakdown.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5] if cat_breakdown else []
    
    # Calculate daily average spending (add 1 to include both start and end dates)
    date_range_days = (df_clean['Date'].max() - df_clean['Date'].min()).days + 1
    daily_avg = total / max(1, date_range_days) if date_range_days > 0 else total
    
    return {
        'total_spending': round(total, 2),
        'average_spending': round(mean, 2),
        'median_spending': round(median, 2),
        'std_deviation': round(std, 2),
        'daily_average': round(daily_avg, 2),
        'category_breakdown': cat_breakdown,
        'category_stats': cat_stats,
        'top_categories': top_categories,
        'anomalies': anomalies,
        'anomaly_count': len(anomalies),
        'anomaly_rate': round(anomaly_rate * 100, 1),
        'trend_slope': round(float(slope), 4),
        'trend_reliability': round(float(r_squared), 3),
        'forecast_7days': round(forecast_7, 2),
        'forecast_30days': round(forecast_30, 2),
        'forecast_90days': round(forecast_90, 2),
        'forecast_method': 'hybrid' if reliability > 0.5 else 'weighted_average',
        'financial_health_score': round(health_score, 1),
        'health_components': {
            'consistency': round(consistency, 1),
            'diversity': round(diversity, 1),
            'anomaly_impact': round(anomaly_impact, 1),
            'trend_stability': round(trend_stability, 1),
            'budget_adherence': round(budget_adherence, 1)
        },
        'spending_velocity': round(velocity, 1),
        'velocity_direction': 'increasing' if velocity > 5 else 'decreasing' if velocity < -5 else 'stable',
        'category_count': num_cats,
        'transaction_count': len(df_clean),
        'expense_count': len(df_expenses),
        'income_count': len(df_income),
        'date_range': {
            'start': df_clean['Date'].min().strftime('%Y-%m-%d'),
            'end': df_clean['Date'].max().strftime('%Y-%m-%d'),
            'days': date_range_days - 1  # Actual difference in days
        },
        'data_quality': {
            'completeness': 100.0,
            'has_negatives': len(df_income) > 0,
            'numeric_issues': 0
        }
    }

def get_empty_analytics():
    """Return safe empty analytics when no data"""
    return {
        'total_spending': 0.0,
        'average_spending': 0.0,
        'median_spending': 0.0,
        'std_deviation': 0.0,
        'daily_average': 0.0,
        'category_breakdown': {},
        'category_stats': {},
        'top_categories': [],
        'anomalies': [],
        'anomaly_count': 0,
        'anomaly_rate': 0.0,
        'trend_slope': 0.0,
        'trend_reliability': 0.0,
        'forecast_7days': 0.0,
        'forecast_30days': 0.0,
        'forecast_90days': 0.0,
        'forecast_method': 'none',
        'financial_health_score': 0.0,
        'health_components': {
            'consistency': 0.0,
            'diversity': 0.0,
            'anomaly_impact': 0.0,
            'trend_stability': 0.0,
            'budget_adherence': 0.0
        },
        'spending_velocity': 0.0,
        'velocity_direction': 'stable',
        'category_count': 0,
        'transaction_count': 0,
        'expense_count': 0,
        'income_count': 0,
        'date_range': {
            'start': 'N/A',
            'end': 'N/A',
            'days': 0
        },
        'data_quality': {
            'completeness': 0.0,
            'has_negatives': False,
            'numeric_issues': 0
        }
    }

# ==================== PDF GENERATION ====================
def generate_pdf_report(df, analytics):
    """
    Generate comprehensive PDF report with all analytics data
    Returns BytesIO object for download
    """
    if not PDF_AVAILABLE:
        return None
    
    # Create BytesIO buffer
    buffer = BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#3b82f6'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Title Page
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("💎 FinanceIQ", title_style))
    elements.append(Paragraph("Financial Analytics Report", heading_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(f"<b>Generated:</b> {report_date}", normal_style))
    elements.append(Paragraph(f"<b>Report Period:</b> {analytics['date_range']['start']} to {analytics['date_range']['end']}", normal_style))
    elements.append(Paragraph(f"<b>Total Transactions:</b> {analytics['transaction_count']:,}", normal_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    elements.append(Paragraph("📊 Executive Summary", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Spending', f"${analytics['total_spending']:,.2f}"],
        ['Average Transaction', f"${analytics['average_spending']:,.2f}"],
        ['Median Transaction', f"${analytics['median_spending']:,.2f}"],
        ['Daily Average', f"${analytics['daily_average']:,.2f}"],
        ['Financial Health Score', f"{analytics['financial_health_score']:.1f}/100"],
        ['Number of Categories', str(analytics['category_count'])],
        ['Spending Velocity', f"{analytics['spending_velocity']:+.1f}%"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')])
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.4*inch))
    
    # Financial Health Score Breakdown
    elements.append(Paragraph("💪 Financial Health Analysis", heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    health_text = f"""Your Financial Health Score is <b>{analytics['financial_health_score']:.1f}/100</b>, 
    indicating {"excellent" if analytics['financial_health_score'] >= 80 else "good" if analytics['financial_health_score'] >= 60 else "moderate" if analytics['financial_health_score'] >= 40 else "needs improvement"} 
    financial management. This score is calculated based on multiple factors including spending consistency, 
    category diversity, anomaly detection, and trend stability."""
    elements.append(Paragraph(health_text, normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    health_components = [
        ['Component', 'Score', 'Weight'],
        ['Consistency', f"{analytics['health_components']['consistency']:.1f}/100", '30%'],
        ['Diversity', f"{analytics['health_components']['diversity']:.1f}/100", '20%'],
        ['Anomaly Impact', f"{analytics['health_components']['anomaly_impact']:.1f}/100", '20%'],
        ['Trend Stability', f"{analytics['health_components']['trend_stability']:.1f}/100", '20%'],
        ['Budget Adherence', f"{analytics['health_components']['budget_adherence']:.1f}/100", '10%']
    ]
    
    health_table = Table(health_components, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecfdf5')])
    ]))
    elements.append(health_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Page Break
    elements.append(PageBreak())
    
    # Category Breakdown
    elements.append(Paragraph("🏷️ Category Breakdown", heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    category_data = [['Category', 'Total Spent', 'Percentage', 'Avg. Transaction']]
    total = analytics['total_spending']
    
    for category, amount in analytics['top_categories'][:10]:  # Top 10 categories
        percentage = (amount / total * 100) if total > 0 else 0
        # Get category stats safely
        avg_amount = analytics.get('category_breakdown', {}).get(category, amount)
        if category in analytics.get('category_stats', {}):
            avg_amount = analytics['category_stats'][category][('Amount', 'mean')]
        else:
            avg_amount = amount  # Fallback
        
        category_data.append([
            category,
            f"${amount:,.2f}",
            f"{percentage:.1f}%",
            f"${avg_amount:,.2f}"
        ])
    
    category_table = Table(category_data, colWidths=[2*inch, 1.3*inch, 1*inch, 1.2*inch])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ff')])
    ]))
    elements.append(category_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Trend Analysis
    elements.append(Paragraph("📈 Trend Analysis", heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    trend_direction = "increasing" if analytics['trend_slope'] > 0 else "decreasing" if analytics['trend_slope'] < 0 else "stable"
    velocity_direction = analytics['velocity_direction']
    
    trend_text = f"""Your spending trend is currently <b>{trend_direction}</b> with a slope of 
    {analytics['trend_slope']:.2f}. The trend reliability score is {analytics['trend_reliability']:.1%}, 
    indicating {"high" if analytics['trend_reliability'] > 0.7 else "moderate" if analytics['trend_reliability'] > 0.4 else "low"} 
    confidence in predictions. Your spending velocity is <b>{velocity_direction}</b> at 
    {analytics['spending_velocity']:+.1f}% compared to the previous period."""
    
    elements.append(Paragraph(trend_text, normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Forecasts
    forecast_data = [
        ['Period', 'Projected Spending', 'Method'],
        ['Next 7 Days', f"${analytics['forecast_7days']:,.2f}", analytics['forecast_method']],
        ['Next 30 Days', f"${analytics['forecast_30days']:,.2f}", analytics['forecast_method']],
        ['Next 90 Days', f"${analytics['forecast_90days']:,.2f}", analytics['forecast_method']]
    ]
    
    forecast_table = Table(forecast_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f59e0b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef3c7')])
    ]))
    elements.append(forecast_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Anomalies
    if analytics['anomaly_count'] > 0:
        elements.append(Paragraph("⚠️ Anomaly Detection", heading_style))
        elements.append(Spacer(1, 0.15*inch))
        
        anomaly_text = f"""Detected <b>{analytics['anomaly_count']}</b> unusual transactions 
        (anomaly rate: {analytics['anomaly_rate']:.1%}). These transactions significantly deviate 
        from your typical spending patterns and may require attention."""
        elements.append(Paragraph(anomaly_text, normal_style))
        elements.append(Spacer(1, 0.2*inch))
    
    # Page Break
    elements.append(PageBreak())
    
    # Transaction Details
    elements.append(Paragraph("📋 Transaction History", heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Show last 20 transactions
    df_sorted = df.sort_values('Date', ascending=False).head(20)
    
    transaction_data = [['Date', 'Category', 'Amount']]
    for _, row in df_sorted.iterrows():
        transaction_data.append([
            row['Date'].strftime('%Y-%m-%d'),
            str(row['Category'])[:30],  # Limit length
            f"${row['Amount']:,.2f}"
        ])
    
    transaction_table = Table(transaction_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch])
    transaction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecfeff')])
    ]))
    elements.append(transaction_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Statistical Summary
    elements.append(Paragraph("📊 Statistical Summary", heading_style))
    elements.append(Spacer(1, 0.15*inch))
    
    stats_data = [
        ['Statistic', 'Value'],
        ['Mean', f"${analytics['average_spending']:,.2f}"],
        ['Median', f"${analytics['median_spending']:,.2f}"],
        ['Standard Deviation', f"${analytics['std_deviation']:,.2f}"],
        ['Minimum Transaction', f"${df['Amount'].min():,.2f}"],
        ['Maximum Transaction', f"${df['Amount'].max():,.2f}"],
        ['Total Days Analyzed', str(analytics['date_range']['days'])],
        ['Average per Day', f"${analytics['daily_average']:,.2f}"]
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef2ff')])
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", footer_style))
    elements.append(Paragraph("<b>FinanceIQ</b> - Advanced Financial Analytics Platform", footer_style))
    elements.append(Paragraph(f"Report Generated on {report_date}", footer_style))
    elements.append(Paragraph("Developed by Syed Sajjad Hussain | 2026", footer_style))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF data
    buffer.seek(0)
    return buffer

# ==================== AI CHATBOT INTELLIGENCE ====================
def is_data_related_question(question):
    """Check if question is related to financial data"""
    data_keywords = [
        'spend', 'money', 'expense', 'budget', 'save', 'cost', 'financial',
        'category', 'transaction', 'payment', 'income', 'balance', 'analytics',
        'forecast', 'predict', 'trend', 'health', 'analysis', 'breakdown',
        'total', 'average', 'how much', 'where', 'what', 'show', 'compare',
        'reduce', 'increase', 'analyze', 'help', 'advice', 'recommend'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in data_keywords)

def get_intelligent_response(question, financial_data, analytics, response_size='detailed'):
    """
    Intelligent AI response system with size options
    Only answers data-related questions
    """
    
    # Check if question is data-related
    if not is_data_related_question(question):
        return """🤖 **FinanceIQ Assistant**

I'm your Financial Analytics AI, specialized in analyzing your spending data and providing financial insights.

I can help you with:
- 💰 Spending analysis and breakdowns
- 📊 Financial health assessment  
- 🎯 Budget optimization strategies
- 📈 Trend analysis and forecasting
- 💡 Personalized saving recommendations

Please ask me something related to your financial data, and I'll provide detailed insights!

**Example questions:**
- "How can I save $500 this month?"
- "What are my biggest expenses?"
- "Analyze my spending patterns"
- "Where am I overspending?"
"""
    
    # Prepare context
    sorted_cats = sorted(
        financial_data['categories'],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Debug output
    print(f"DEBUG: ultra_ai exists: {ultra_ai is not None}")
    print(f"DEBUG: ultra_ai.anthropic_client exists: {ultra_ai.anthropic_client is not None if ultra_ai else False}")
    
    # Get AI response with proper error handling
    if ultra_ai and ultra_ai.anthropic_client:
        try:
            print(f"\n{'='*60}")
            print(f"🤖 PROCESSING NEW QUESTION")
            print(f"Question: {question}")
            print(f"Mode: {response_size}")
            print(f"{'='*60}\n")
            
            response = ultra_ai.get_smart_response(
                question,
                financial_data,
                analytics,
                mode='smart'
            )
            
            print(f"\n✅ Successfully generated AI response: {len(response)} characters\n")
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ ERROR calling Claude API: {error_msg}\n")
            import traceback
            traceback.print_exc()
            
            # Return error info to user
            return f"""⚠️ **AI Service Error**
            
The AI service encountered an issue: {error_msg}

Please check:
1. Your Anthropic API key is correctly set in `.env` file
2. Your API key has sufficient credits
3. You have internet connection

Using fallback analysis mode..."""
    else:
        print("\n⚠️ WARNING: No AI client available - using fallback responses\n")
        print(f"API_KEY exists: {bool(API_KEY)}")
        print(f"ultra_ai exists: {ultra_ai is not None}")
        if ultra_ai:
            print(f"anthropic_client exists: {ultra_ai.anthropic_client is not None}")
    
    # Intelligent fallback responses
    top_cat, top_amt = sorted_cats[0] if sorted_cats else ("N/A", 0)
    
    if response_size == 'concise':
        return f"""**Quick Answer:**

Top expense: {top_cat} (${top_amt:,.2f})
Health Score: {analytics['financial_health_score']:.0f}/100

💡 Reduce {top_cat} by 15% to save ${top_amt * 0.15:,.2f}/month!"""
    
    elif response_size == 'comprehensive':
        return f"""**Comprehensive Financial Analysis:**

📊 **Overview:**
- Total Spending: ${financial_data['total']:,.2f}
- Average Transaction: ${financial_data['avg']:.2f}
- Financial Health: {analytics['financial_health_score']:.0f}/100
- Spending Trend: {"Increasing" if analytics['trend_slope'] > 0 else "Decreasing"}

💰 **Top Categories:**
{chr(10).join([f"{i+1}. {cat}: ${amt:,.2f} ({amt/financial_data['total']*100:.1f}%)" for i, (cat, amt) in enumerate(sorted_cats[:5])])}

🎯 **Optimization Strategy:**
1. **Primary Target:** {top_cat} (${top_amt:,.2f})
   - Reduce by 10%: Save ${top_amt * 0.1:,.2f}/month
   - Reduce by 20%: Save ${top_amt * 0.2:,.2f}/month

2. **Set Budget Limits:**
   - {top_cat}: ${top_amt * 0.85:,.2f}/month
   - Overall: ${financial_data['total'] * 0.9:,.2f}/month

3. **Track Daily:** Monitor {top_cat} expenses closely

📈 **30-Day Forecast:** ${analytics['forecast_30days']:,.2f}

💡 **Recommendation:**
Focus on your top 2 categories for maximum impact. Small consistent changes lead to big savings!"""
    
    else:  # detailed (default)
        return f"""**Financial Insights:**

Your biggest expense is **{top_cat}** at ${top_amt:,.2f}/month ({top_amt/financial_data['total']*100:.1f}% of spending).

**Financial Health:** {analytics['financial_health_score']:.0f}/100

**Savings Opportunity:**
- Reduce {top_cat} by 15%: **${top_amt * 0.15:,.2f}/month**
- Annual savings: **${top_amt * 0.15 * 12:,.2f}**

**Action Plan:**
1. Set monthly budget: ${top_amt * 0.85:,.2f} for {top_cat}
2. Track expenses daily
3. Review weekly progress

**Forecast:** Next month estimated at ${analytics['forecast_30days']:,.2f}

💡 Small changes today = Big savings tomorrow!"""

# ==================== PREMIUM UI STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0 2rem;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
        animation: gradient-shift 8s ease infinite;
        letter-spacing: -0.02em;
    }
    
    @keyframes gradient-shift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
    }
    
    .tagline {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .developer-credit {
        text-align: center;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Perfect Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.875rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: 2px solid transparent;
        text-align: center;
        line-height: 1.5;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Perfect Card Alignment */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 1.5rem 1.75rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        height: 100%;
        min-height: 140px;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        overflow: visible !important;
        box-sizing: border-box;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        div[data-testid="stMetric"] {
            padding: 1.25rem 1.5rem;
            min-height: 120px;
        }
        
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        .main-header {
            font-size: 2.5rem !important;
        }
        
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* Small Mobile */
    @media (max-width: 480px) {
        div[data-testid="stMetric"] {
            padding: 1rem 1.25rem;
            min-height: 110px;
        }
        
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
        
        div[data-testid="stMetric"] label {
            font-size: 0.8rem !important;
        }
        
        .main-header {
            font-size: 2rem !important;
        }
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    div[data-testid="stMetric"] label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
        margin-bottom: 0.75rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        width: 100% !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        line-height: 1.4 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
        max-width: 100% !important;
        width: 100% !important;
        display: block !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
        max-width: 100% !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        margin-top: 0.5rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        max-width: 100% !important;
    }
    
    /* Perfect Info/Warning/Success Boxes */
    .stAlert {
        border-radius: 12px;
        border: 1px solid;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    
    div[data-baseweb="notification"] {
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        border-width: 1px !important;
    }
    
    /* Perfect Chat Container */
    .chat-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    /* Perfect Message Bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.25rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.75rem 0;
        animation: slideInRight 0.4s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 1rem !important;
        line-height: 1.6 !important;
        word-wrap: break-word !important;
        white-space: pre-wrap !important;
        overflow-wrap: break-word !important;
    }
    
    .user-message * {
        color: white !important;
        white-space: pre-wrap !important;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        color: #1f2937 !important;
        padding: 1.25rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.75rem 0;
        animation: slideInLeft 0.4s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #d1d5db;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        word-wrap: break-word !important;
        white-space: pre-wrap !important;
        overflow-wrap: break-word !important;
    }
    
    .ai-message * {
        color: #1f2937 !important;
        white-space: pre-wrap !important;
    }
    
    .ai-message p, .user-message p {
        margin: 0.5rem 0 !important;
        white-space: pre-wrap !important;
    }
    
    .ai-message ul, .ai-message ol {
        margin-left: 1.5rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .ai-message li {
        margin: 0.25rem 0 !important;
    }
    
    .ai-message strong, .user-message strong {
        font-weight: 700 !important;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Perfect Input Fields */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        padding: 0.875rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Perfect File Uploader */
    .stFileUploader {
        border-radius: 12px;
    }
    
    div[data-testid="stFileUploadDropzone"] {
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
        padding: 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* Perfect Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Perfect Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
    }
    
    /* Perfect Expander */
    .streamlit-expanderHeader {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e5e7eb;
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 1.25rem;
    }
    
    /* Perfect Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    /* Fix ALL text truncation issues */
    .stDataFrame table {
        white-space: normal !important;
    }
    
    .stDataFrame td, .stDataFrame th {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-wrap: break-word !important;
    }
    
    /* Fix markdown text truncation */
    .stMarkdown p, .stMarkdown div {
        white-space: pre-wrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-wrap: break-word !important;
    }
    
    /* Fix all text elements - REMOVE overly broad rule */
    .stMarkdown {
        overflow: visible !important;
    }
    
    /* Ensure numbers display fully */
    .element-container {
        overflow: visible !important;
    }
    
    /* Perfect Columns Alignment */
    .row-widget.stHorizontal {
        gap: 1rem;
        flex-wrap: wrap !important;
    }
    
    /* Responsive column layout */
    @media (max-width: 768px) {
        .row-widget.stHorizontal > div {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .row-widget.stHorizontal > div {
            flex: 1 1 48% !important;
            min-width: 48% !important;
        }
    }
    
    @media (min-width: 1025px) {
        .row-widget.stHorizontal > div {
            flex: 1 1 auto !important;
        }
    }
    
    /* Perfect Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #374151;
    }
    
    /* Perfect Radio Buttons */
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 0.875rem 1.25rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f9fafb;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Perfect Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Remove extra padding */
    .element-container {
        margin: 0;
    }
    
    /* Perfect Header Spacing */
    h1, h2, h3 {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.3 !important;
    }
    
    h1 {
        font-weight: 700 !important;
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-weight: 600 !important;
        font-size: 2rem !important;
    }
    
    h3 {
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="main-header">💎 FinanceIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">AI-Powered Financial Intelligence • Smooth Analytics • Instant Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="developer-credit">Developed by <strong>Syed Sajjad Hussain</strong> • 2026</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🎯 FinanceIQ")
    
    # Get current page from session state or radio
    page = st.radio(
        "Navigation",
        ["📤 Upload Data", "📊 Analytics Dashboard", "🤖 Smart Analytics", "🎨 Visual Insights"],
        index=["📤 Upload Data", "📊 Analytics Dashboard", "🤖 Smart Analytics", "🎨 Visual Insights"].index(st.session_state.page),
        label_visibility="collapsed",
        key="main_navigation"
    )
    st.session_state.page = page
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.analytics_data:
        st.markdown("### 📊 Quick Stats")
        a = st.session_state.analytics_data
        st.metric("Transactions", f"{a['transaction_count']:,}")
        st.metric("Health Score", f"{a['financial_health_score']:.0f}/100")
        st.metric("Categories", a['category_count'])
    
    st.markdown("---")
    st.caption("©FinanceIQ \n v1.0.0")

# ==================== PAGE: UPLOAD ====================
if page == "📤 Upload Data":
    st.header("📤 Upload Financial Data")
    
    # Clean, centered upload section
    uploaded = st.file_uploader(
        "📁 Choose your CSV file",
        type=['csv'],
        help="Upload any CSV file - our AI will detect dates, amounts, and categories"
    )
    
    st.markdown("<div style='text-align: center; margin: 20px 0;'><p>OR</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        use_sample = st.button("📦 Use Sample Data", use_container_width=True, type="primary", help="Load demo financial data")
    
    if use_sample:
        uploaded = "data/sample_expenses.csv"
        st.success("✅ Sample data loaded! Scroll down to see it.")
        time.sleep(0.5)
    
    if uploaded:
        try:
            # Smooth loading animation
            with st.spinner("🔄 Processing your data..."):
                time.sleep(0.5)  # Smooth UX
                
                df_raw = pd.read_csv(uploaded) if isinstance(uploaded, str) else pd.read_csv(uploaded)
                
                st.success(f"✅ Loaded {len(df_raw):,} rows with {len(df_raw.columns)} columns")
                
                # Smart column detection
                with st.expander("🧠 AI Column Detection", expanded=True):
                    st.markdown("**Detected columns:**")
                    
                    # Auto-detect or manual mapping
                    col1, col2, col3 = st.columns(3)
                    
                    # Detect date column
                    date_candidates = []
                    for col in df_raw.columns:
                        if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():
                            date_candidates.append(col)
                        else:
                            # Try to parse as date
                            try:
                                pd.to_datetime(df_raw[col].head(10), errors='coerce', format='mixed')
                                if pd.to_datetime(df_raw[col], errors='coerce', format='mixed').notna().sum() > len(df_raw) * 0.5:
                                    date_candidates.append(col)
                            except:
                                pass
                    
                    # Detect amount/numeric column
                    amount_candidates = []
                    for col in df_raw.columns:
                        if any(word in col.lower() for word in ['amount', 'price', 'cost', 'value', 'total', 'sum', 'expense', 'spend']):
                            amount_candidates.append(col)
                        elif df_raw[col].dtype in ['int64', 'float64']:
                            amount_candidates.append(col)
                    
                    # Detect category column
                    category_candidates = []
                    for col in df_raw.columns:
                        if any(word in col.lower() for word in ['category', 'type', 'class', 'group', 'name', 'description']):
                            category_candidates.append(col)
                        elif df_raw[col].dtype == 'object' and col not in date_candidates:
                            # Check if it has reasonable number of unique values
                            if 2 <= df_raw[col].nunique() <= len(df_raw) * 0.5:
                                category_candidates.append(col)
                    
                    with col1:
                        date_col = st.selectbox(
                            "📅 Date Column",
                            options=date_candidates + [c for c in df_raw.columns if c not in date_candidates],
                            index=0 if date_candidates else 0,
                            help="Select the column containing dates"
                        )
                    
                    with col2:
                        amount_col = st.selectbox(
                            "💰 Amount Column",
                            options=amount_candidates + [c for c in df_raw.columns if c not in amount_candidates],
                            index=0 if amount_candidates else 0,
                            help="Select the column containing amounts/values"
                        )
                    
                    with col3:
                        category_col = st.selectbox(
                            "🏷️ Category Column",
                            options=category_candidates + [c for c in df_raw.columns if c not in category_candidates],
                            index=0 if category_candidates else 0,
                            help="Select the column for grouping/categories"
                        )
                    
                    # Show auto-detection confidence
                    confidence_msg = "✨ **AI Confidence:** "
                    if date_col in date_candidates and amount_col in amount_candidates and category_col in category_candidates:
                        confidence_msg += "🟢 High - All columns auto-detected!"
                    elif (date_col in date_candidates or amount_col in amount_candidates) and category_col in category_candidates:
                        confidence_msg += "🟡 Medium - Please verify selections"
                    else:
                        confidence_msg += "🟠 Low - Manual selection recommended"
                    
                    st.info(confidence_msg)
                
                # Map and clean data
                df = pd.DataFrame({
                    'Date': df_raw[date_col],
                    'Category': df_raw[category_col],
                    'Amount': df_raw[amount_col]
                })
                
                # Clean and prepare
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
                df['Category'] = df['Category'].astype(str)
                df = df.dropna()
                df = df.sort_values('Date')
                
                st.session_state.df = df
                
                # Success message with smooth transition
                st.success(f"✅ Successfully loaded {len(df):,} transactions!")
                
                # Preview with smooth reveal
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(
                        df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Quick metrics
                st.markdown("### 📊 Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{len(df):,}")
                with col2:
                    st.metric("Total Amount", f"${df['Amount'].sum():,.2f}")
                with col3:
                    st.metric("Average", f"${df['Amount'].mean():.2f}")
                with col4:
                    st.metric("Categories", f"{df['Category'].nunique()}")
                
                st.markdown("---")
                
                # Analyze button with clear guidance
                st.info("👇 **Ready to analyze?** Click the button below to run advanced AI analytics on your data!")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🧠 Run Advanced Analytics", type="primary", use_container_width=True, key="run_analytics_btn"):
                        with st.spinner("🔬 Performing deep analysis..."):
                            # Smooth progress
                            progress = st.progress(0)
                            for i in range(100):
                                time.sleep(0.01)
                                progress.progress(i + 1)
                            
                            # Run analytics
                            analytics = perform_advanced_analytics(df)
                            st.session_state.analytics_data = analytics
                            st.session_state.analysis_done = True
                            
                            # Save to database with lowercase column names
                            conn = sqlite3.connect(DB_PATH)
                            df_save = df.copy()
                            df_save.columns = [col.lower() for col in df_save.columns]  # Lowercase for consistency
                            df_save.to_sql('expenses', conn, if_exists='replace', index=False)
                            conn.close()
                            
                            progress.empty()
                            
                            # Success celebration
                            st.success("✅ Analysis Complete!")
                            st.balloons()
                            
                            # Show key results
                            st.markdown("### 🎯 Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                health = analytics['financial_health_score']
                                health_emoji = "🟢" if health >= 75 else "🟡" if health >= 50 else "🔴"
                                st.metric(
                                    "Financial Health",
                                    f"{health:.0f}/100 {health_emoji}",
                                    delta="Excellent" if health >= 75 else "Good" if health >= 50 else "Needs Attention"
                                )
                            with col2:
                                st.metric(
                                    "30-Day Forecast",
                                    f"${analytics['forecast_30days']:,.2f}"
                                )
                            with col3:
                                st.metric("Anomalies Found", analytics['anomaly_count'], delta=f"{analytics['anomaly_rate']}%" if analytics['anomaly_rate'] > 0 else "Clean")
                            
                            # Additional insights
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Daily Average", f"${analytics.get('daily_average', 0):,.2f}")
                            with col2:
                                trend_rel = analytics.get('trend_reliability', 0)
                                st.metric("Forecast Confidence", f"{trend_rel * 100:.0f}%", delta="High" if trend_rel > 0.7 else "Medium" if trend_rel > 0.4 else "Low")
                            
                            st.markdown("---")
                            st.success("💡 Analysis complete! Choose where to go next:")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("📊 View Dashboard", use_container_width=True, type="primary"):
                                    st.session_state.page = "📊 Analytics Dashboard"
                                    st.rerun()
                            with col2:
                                if st.button("🤖 Chat with AI", use_container_width=True, type="primary"):
                                    st.session_state.page = "🤖 Smart Analytics"
                                    st.rerun()
                            with col3:
                                if st.button("🎨 See Visuals", use_container_width=True, type="primary"):
                                    st.session_state.page = "🎨 Visual Insights"
                                    st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 **Tip:** Upload any CSV with date, amount, and category-like columns. Our AI will detect them!")
    else:
        st.info("👆 Upload any CSV file - our AI will intelligently analyze it!")
        
        # Show capabilities
        st.markdown("### 🎯 What Makes This Smart?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""**📊 Flexible Analysis**
            - Any CSV structure
            - Auto-detects columns
            - Works with messy data""")
        
        with col2:
            st.markdown("""**🧠 AI-Powered**
            - Intelligent mapping
            - Pattern recognition
            - Smart cleaning""")
        
        with col3:
            st.markdown("""**⚡ Lightning Fast**
            - Instant detection
            - Real-time preview
            - Smooth experience""")
        
        st.markdown("---")
        st.markdown("### 💡 Example CSV Formats We Support")
        
        tab1, tab2, tab3 = st.tabs(["Format 1", "Format 2", "Format 3"])
        
        with tab1:
            st.caption("Standard financial format")
            sample1 = pd.DataFrame({
                'Date': ['2024-01-01', '2024-01-02'],
                'Category': ['Groceries', 'Transport'],
                'Amount': [150.50, 45.00]
            })
            st.dataframe(sample1, use_container_width=True, hide_index=True)
        
        with tab2:
            st.caption("E-commerce/Sales format")
            sample2 = pd.DataFrame({
                'Transaction_Date': ['2024-01-01', '2024-01-02'],
                'Product_Type': ['Electronics', 'Clothing'],
                'Total_Price': [299.99, 79.50]
            })
            st.dataframe(sample2, use_container_width=True, hide_index=True)
        
        with tab3:
            st.caption("Custom business format")
            sample3 = pd.DataFrame({
                'Timestamp': ['2024-01-01 10:30', '2024-01-02 15:45'],
                'Department': ['Marketing', 'Sales'],
                'Cost': [1250.00, 890.00]
            })
            st.dataframe(sample3, use_container_width=True, hide_index=True)

# ==================== PAGE: DASHBOARD ====================
elif page == "📊 Analytics Dashboard":
    st.header("📊 Financial Analytics Dashboard")
    
    if not st.session_state.analysis_done or st.session_state.df is None:
        st.warning("⚠️ No data analyzed yet. Please upload and analyze your data first.")
        if st.button("📤 Go to Upload Page", use_container_width=True, type="primary"):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    else:
        df = st.session_state.df
        a = st.session_state.analytics_data
        
        # PDF Download Section
        st.markdown("### 📄 Export Report")
        
        with st.expander("ℹ️ What's included in the PDF Report?", expanded=False):
            st.markdown("""
            **Comprehensive PDF Report includes:**
            
            📊 **Executive Summary**
            - Total spending, averages, and key metrics
            - Financial health score overview
            
            💪 **Financial Health Analysis**
            - Detailed breakdown of health components
            - Consistency, diversity, and trend stability scores
            
            🏷️ **Category Breakdown**
            - Top 10 spending categories
            - Percentage distribution and transaction counts
            
            📈 **Trend Analysis & Forecasts**
            - Spending trend direction and reliability
            - 7-day, 30-day, and 90-day projections
            
            ⚠️ **Anomaly Detection**
            - Unusual transaction identification
            - Anomaly rate analysis
            
            📋 **Transaction History**
            - Recent 20 transactions with details
            - Date, category, and amount information
            
            📊 **Statistical Summary**
            - Complete statistical analysis
            - Min, max, mean, median, and standard deviation
            
            **All data is professionally formatted with tables and charts!**
            """)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            if PDF_AVAILABLE:
                if st.button("📥 Generate PDF Report", type="primary", use_container_width=True, key="generate_pdf_btn"):
                    with st.spinner("🔄 Generating comprehensive PDF report..."):
                        try:
                            # Generate PDF
                            pdf_buffer = generate_pdf_report(df, a)
                            
                            if pdf_buffer:
                                st.session_state.pdf_buffer = pdf_buffer
                                st.session_state.pdf_ready = True
                                st.success("✅ PDF generated! Download button below:")
                                st.rerun()
                            else:
                                st.error("❌ Failed to generate PDF report")
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
                            st.info("💡 Please ensure all required packages are installed: `pip install reportlab matplotlib`")
                
                # Show download button if PDF is ready
                if st.session_state.get('pdf_ready', False):
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=st.session_state.pdf_buffer,
                        file_name=f"FinanceIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="download_pdf_btn"
                    )
                    st.info("💡 **Tip:** The PDF includes all your analytics, beautifully formatted!")
            else:
                st.warning("📦 Install reportlab to enable PDF export: `pip install reportlab matplotlib`")
        
        st.markdown("---")
        
        # Key metrics with smooth animation
        st.markdown("### 💎 Financial Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total = a['total_spending']
            # Format large numbers properly to avoid truncation
            st.metric("Total Spending", f"${total:,.2f}")
        with col2:
            avg = a['average_spending']
            st.metric("Average", f"${avg:,.2f}")
        with col3:
            health = a['financial_health_score']
            st.metric("Health Score", f"{health:.0f}/100")
        with col4:
            st.metric("Categories", a['category_count'])
        with col5:
            velocity = a['spending_velocity']
            vel_dir = a.get('velocity_direction', 'stable')
            vel_emoji = "📈" if vel_dir == 'increasing' else "📉" if vel_dir == 'decreasing' else "➡️"
            st.metric("Velocity", f"{velocity:+.1f}%", delta=vel_dir.title(), delta_color="inverse")
        
        # Data quality indicator
        if a.get('data_quality', {}).get('has_negatives', False):
            st.info("ℹ️ **Note:** Your data includes negative amounts (refunds/income). Analysis focuses on expenses (positive amounts).")
        
        st.markdown("---")
        
        # Main visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Deep Dive"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 Spending Distribution")
                sorted_cats = sorted(a['category_breakdown'].items(), key=lambda x: x[1], reverse=True)
                
                fig = go.Figure([go.Pie(
                    labels=[c[0] for c in sorted_cats],
                    values=[c[1] for c in sorted_cats],
                    hole=0.5,
                    marker=dict(
                        colors=px.colors.qualitative.Set3,
                        line=dict(color='white', width=2)
                    ),
                    textposition='auto',
                    textinfo='label+percent'
                )])
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📅 Spending Timeline")
                daily = df.groupby('Date')['Amount'].sum().reset_index()
                
                fig = go.Figure([go.Scatter(
                    x=daily['Date'],
                    y=daily['Amount'],
                    mode='lines+markers',
                    name='Daily Spending',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, color='#764ba2'),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                )])
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Amount ($)",
                    margin=dict(t=20, b=20, l=20, r=20),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Category Comparison")
                fig = go.Figure([go.Bar(
                    x=[c[0] for c in sorted_cats],
                    y=[c[1] for c in sorted_cats],
                    marker=dict(
                        color=[c[1] for c in sorted_cats],
                        colorscale='Viridis',
                        showscale=False
                    )
                )])
                fig.update_layout(
                    height=400,
                    xaxis_title="Category",
                    yaxis_title="Amount ($)",
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔮 Forecast")
                forecast_data = pd.DataFrame({
                    'Period': ['Current', '7 Days', '30 Days', '90 Days'],
                    'Amount': [
                        a['total_spending'],
                        a['forecast_7days'],
                        a['forecast_30days'],
                        a['forecast_90days']
                    ]
                })
                
                fig = go.Figure([go.Scatter(
                    x=forecast_data['Period'],
                    y=forecast_data['Amount'],
                    mode='lines+markers',
                    line=dict(color='#764ba2', width=3),
                    marker=dict(size=12)
                )])
                fig.update_layout(
                    height=400,
                    yaxis_title="Projected Amount ($)",
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 📋 Detailed Category Breakdown")
            
            breakdown_df = pd.DataFrame([
                {
                    'Category': cat,
                    'Amount': f"${amt:,.2f}",
                    'Percentage': f"{(amt/a['total_spending']*100):.1f}%",
                    'Transactions': df[df['Category'] == cat].shape[0]
                }
                for cat, amt in sorted_cats
            ])
            
            st.dataframe(
                breakdown_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Amount": st.column_config.TextColumn("Total Amount", width="small"),
                    "Percentage": st.column_config.TextColumn("% of Total", width="small"),
                    "Transactions": st.column_config.NumberColumn("Count", width="small")
                }
            )
            
            # Health components
            st.markdown("#### 🏥 Health Score Breakdown")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            components = a['health_components']
            with col1:
                st.metric("Consistency", f"{components['consistency']:.0f}/100")
            with col2:
                st.metric("Diversity", f"{components['diversity']:.0f}/100")
            with col3:
                st.metric("Anomaly Impact", f"{components['anomaly_impact']:.0f}/100")
            with col4:
                st.metric("Trend Stability", f"{components['trend_stability']:.0f}/100")
            with col5:
                st.metric("Budget Adherence", f"{components.get('budget_adherence', 0):.0f}/100")
            
            # Data quality metrics
            st.markdown("#### 📊 Data Quality")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", a['transaction_count'])
            with col2:
                st.metric("Expenses", a.get('expense_count', a['transaction_count']))
            with col3:
                st.metric("Date Range", f"{a['date_range'].get('days', 0)} days")

# ==================== PAGE: AI ASSISTANT ====================
elif page == "🤖 Smart Analytics":
    st.header("🤖 Smart Analytics AI")
    
    if not st.session_state.analysis_done:
        st.warning("⚠️ Please analyze your data first to chat with the AI assistant.")
        if st.button("📤 Go to Upload Page", use_container_width=True, type="primary"):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    else:
        # Welcome message
        if st.session_state.show_welcome:
            st.info("""
👋 **Welcome to Smart Analytics!**

I'm your intelligent financial advisor, powered by advanced AI. I can analyze your spending patterns and provide personalized recommendations.

**I can help you with:**
- 💰 Budget optimization
- 📊 Spending analysis  
- 🎯 Savings strategies
- 📈 Trend forecasting
- 💡 Financial advice

**Just ask me anything about your financial data!**
            """)
            st.session_state.show_welcome = False
        
        # Response size selector
        st.markdown("### ⚙️ Response Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Concise", use_container_width=True, 
                        type="primary" if st.session_state.response_mode == 'concise' else "secondary"):
                st.session_state.response_mode = 'concise'
                st.rerun()
        with col2:
            if st.button("📝 Detailed", use_container_width=True,
                        type="primary" if st.session_state.response_mode == 'detailed' else "secondary"):
                st.session_state.response_mode = 'detailed'
                st.rerun()
        with col3:
            if st.button("📄 Comprehensive", use_container_width=True,
                        type="primary" if st.session_state.response_mode == 'comprehensive' else "secondary"):
                st.session_state.response_mode = 'comprehensive'
                st.rerun()
        
        st.markdown(f"<div style='text-align: center; color: #888; margin-top: 10px;'><small>Current mode: <strong>{st.session_state.response_mode.title()}</strong></small></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chat interface
        st.markdown("### 💬 Chat")
        st.markdown("<br>", unsafe_allow_html=True)
        
        question = st.text_input(
            "Ask me anything:",
            placeholder="e.g., How can I save $500 this month?",
            key="chat_input",
            label_visibility="visible"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ask_button = st.button("🚀 Ask AI", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
        with col3:
            st.empty()  # Spacing
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and question:
            with st.spinner("� AI is analyzing your question..."):
                # Brief delay for smooth UX
                time.sleep(0.3)
                
                a = st.session_state.analytics_data
                df = st.session_state.df
                
                sorted_cats = sorted(
                    a['category_breakdown'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                financial_data = {
                    'count': len(df),
                    'total': a['total_spending'],
                    'avg': a['average_spending'],
                    'categories': sorted_cats
                }
                
                # Show processing indicator
                status_placeholder = st.empty()
                status_placeholder.info("🤖 Generating intelligent response...")
                
                response = get_intelligent_response(
                    question,
                    financial_data,
                    a,
                    st.session_state.response_mode
                )
                
                status_placeholder.empty()
                
                # Store in chat history with unique timestamp
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'mode': st.session_state.response_mode
                })
                
                # Clear the input by rerunning
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Conversation History")
            
            # Display in reverse order (most recent first)
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                with st.container():
                    # User message
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 15px; border-radius: 15px; margin: 10px 0;'>
                        <strong style='color: white;'>👤 You ({chat["timestamp"]}):</strong><br>
                        <span style='color: white; font-size: 1.1em;'>{chat["question"]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI response
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 20px; border-radius: 15px; margin: 10px 0;'>
                        <strong style='color: white;'>🤖 AI Assistant ({chat.get("mode", "detailed").title()}):</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display the actual response content
                    st.markdown(chat["answer"])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Suggested questions
        st.markdown("---")
        st.markdown("### 💡 Suggested Questions")
        
        suggestions = [
            "How can I reduce my spending by 20%?",
            "What are my top 3 expense categories?",
            "Analyze my financial health",
            "Where am I overspending?",
            "Create a budget plan for me",
            "What's my spending trend?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                st.markdown(f"<div style='padding: 10px; background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); border-radius: 8px; margin: 5px 0;'><small>💡 {suggestion}</small></div>", unsafe_allow_html=True)

# ==================== PAGE: VISUALIZATIONS ====================
elif page == "🎨 Visual Insights":
    st.header("🎨 Advanced Visual Insights")
    
    if not st.session_state.analysis_done or st.session_state.df is None:
        st.warning("⚠️ No data to visualize. Please upload and analyze data first.")
        if st.button("📤 Go to Upload Page", use_container_width=True, type="primary"):
            st.session_state.page = "📤 Upload Data"
            st.rerun()
    else:
        df = st.session_state.df
        a = st.session_state.analytics_data
        
        # 3D Visualization
        if smart_viz:
            st.markdown("### 🎯 3D Category Analysis")
            try:
                fig = smart_viz.create_3d_category_chart(a['category_breakdown'])
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("3D visualization not available. Showing 2D alternative.")
        
        # Sunburst
        st.markdown("### ☀️ Hierarchical View")
        sorted_cats = sorted(a['category_breakdown'].items(), key=lambda x: x[1], reverse=True)
        
        fig = go.Figure([go.Sunburst(
            labels=[c[0] for c in sorted_cats] + ["Total"],
            parents=["Total"] * len(sorted_cats) + [""],
            values=[c[1] for c in sorted_cats] + [a['total_spending']],
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent entry'
        )])
        fig.update_layout(height=600, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.markdown("### 📅 Spending Heatmap")
        df_copy = df.copy()
        df_copy['DayOfWeek'] = pd.to_datetime(df_copy['Date']).dt.day_name()
        df_copy['Week'] = pd.to_datetime(df_copy['Date']).dt.isocalendar().week
        
        heatmap_data = df_copy.groupby(['Week', 'DayOfWeek'])['Amount'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Week', values='Amount').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
        
        fig = go.Figure([go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='RdYlGn_r',
            text=heatmap_pivot.values,
            texttemplate='$%{text:.0f}',
            textfont={"size": 10},
            colorbar=dict(title="Amount ($)")
        )])
        fig.update_layout(
            height=400,
            xaxis_title="Week Number",
            yaxis_title="Day of Week",
            margin=dict(t=20, b=20, l=80, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6b7280;'>
    <strong>FinanceIQ</strong> - Advanced Financial Analytics Platform<br>
    Developed by <strong>Syed Sajjad Hussain</strong> | 2026<br>
    <small>Optimized for Modern Data Visualization</small>
</div>
""", unsafe_allow_html=True)


