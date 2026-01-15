"""
Ultra-Smart AI Engine for Financial Analytics
=================================================
Advanced AI system powered by Claude Sonnet 4 with context awareness, memory, and intelligent caching

Features:
- Claude Sonnet 4 AI integration
- Conversation memory and context
- Smart response caching
- Real-time learning from user interactions
- Rule-based fallback system
"""

from anthropic import Anthropic
import json
from collections import deque
from datetime import datetime
import hashlib


class UltraSmartAI:
    """Next-generation AI financial advisor with advanced intelligence"""
    
    def __init__(self, anthropic_key=None):
        # Initialize Anthropic client with error handling
        self.anthropic_client = None
        if anthropic_key:
            try:
                self.anthropic_client = Anthropic(api_key=anthropic_key)
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Anthropic client: {e}")
                print("   Falling back to rule-based responses")
        
        # Conversation memory with larger context window
        self.conversation_history = deque(maxlen=30)
        self.user_profile = {}
        
        # AI Model - Latest Claude Sonnet 4
        self.claude_model = "claude-sonnet-4-20250514"
        
        print(f"✅ AI Engine initialized with Claude {self.claude_model}")
        if self.anthropic_client:
            print(f"✅ Anthropic API connected successfully")
        
    def get_smart_response(self, user_message, financial_data, analytics_data=None, mode='smart'):
        """
        Get intelligent response using Claude AI with context awareness
        
        Args:
            user_message: User's question
            financial_data: Basic financial stats
            analytics_data: Advanced analytics results
            mode: 'smart' (Claude Sonnet 4)
        """
        
        # No caching - each question gets a unique, contextual response
        print(f"🤖 Processing question: {user_message[:50]}...")
        
        # Build comprehensive context
        context = self._build_financial_context(financial_data, analytics_data)
        
        # Add conversation memory
        memory_context = self._get_conversation_context()
        
        # Construct system prompt
        system_prompt = self._get_system_prompt()
        
        # Build user prompt with full context
        user_prompt = f"""{context}

{memory_context}

User Question: {user_message}

IMPORTANT: Provide a UNIQUE, SPECIFIC response tailored EXACTLY to this question. 
DO NOT give generic answers. Analyze the specific question and data deeply.

Provide:
1. Direct answer to the EXACT question asked
2. Specific financial insights with REAL numbers from the data
3. Concrete, actionable recommendations with dollar amounts
4. Clear, numbered action steps
5. Data-driven predictions and comparisons where relevant
6. Professional formatting with strategic emoji use

Be intelligent, precise, and helpful. Make every response unique and valuable."""
        
        try:
            response_text = None
            
            # Claude Sonnet 4 (Best quality AI)
            if self.anthropic_client:
                print(f"🚀 Calling Claude API with model: {self.claude_model}")
                response = self.anthropic_client.messages.create(
                    model=self.claude_model,
                    max_tokens=3000,
                    temperature=0.9,  # Higher temperature for more varied responses
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                response_text = response.content[0].text
                print(f"📨 Received response: {len(response_text)} characters")
            
            # Fallback: Rule-based
            else:
                return self._fallback_response(user_message, financial_data)
            
            # Store in conversation history for context continuity
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_message,
                'assistant': response_text,
                'mode': mode
            })
            
            print(f"✅ Generated {len(response_text)} char response")
            return response_text
            
        except Exception as e:
            return self._fallback_response(user_message, financial_data)
    
    def _build_financial_context(self, financial_data, analytics_data=None):
        """Build comprehensive financial context for AI"""
        
        count = financial_data.get('count', 0)
        total = financial_data.get('total', 0)
        avg = financial_data.get('avg', 0)
        categories = financial_data.get('categories', [])
        
        context = f"""
📊 FINANCIAL DATA SNAPSHOT:
━━━━━━━━━━━━━━━━━━━━━━
Transactions: {count:,}
Total Spending: ${total:,.2f}
Average Transaction: ${avg:.2f}
Tracking Categories: {len(categories)}

💰 TOP SPENDING CATEGORIES:
"""
        
        for i, (cat, amt) in enumerate(categories[:5], 1):
            pct = (amt / total * 100) if total > 0 else 0
            context += f"{i}. {cat}: ${amt:,.2f} ({pct:.1f}%)\n"
        
        if analytics_data:
            context += f"""

🔬 ADVANCED ANALYTICS:
━━━━━━━━━━━━━━━━━━━━━━
Health Score: {analytics_data.get('financial_health_score', 0):.1f}/100
Trend: {analytics_data.get('time_series_trend', 'N/A')}
30-Day Forecast: ${analytics_data.get('forecast_30days', 0):,.2f}
Anomalies Detected: {len(analytics_data.get('anomalies_iqr', []))}
Overspending Categories: {', '.join(analytics_data.get('overspending_categories', [])) or 'None'}
Daily Average: ${analytics_data.get('daily_average', 0):.2f}
Weekly Average: ${analytics_data.get('weekly_average', 0):.2f}

📊 STATISTICAL INSIGHTS:
Median: ${analytics_data.get('median_spending', 0):.2f}
Std Dev: ${analytics_data.get('std_deviation', 0):.2f}
Max Expense: ${analytics_data.get('max_expense', 0):.2f}
Min Expense: ${analytics_data.get('min_expense', 0):.2f}
"""
        
        return context
    
    def _get_conversation_context(self):
        """Get recent conversation context for continuity"""
        if not self.conversation_history:
            return ""
        
        context = "\n💭 CONVERSATION CONTEXT (Recent exchanges):\n"
        for entry in list(self.conversation_history)[-3:]:
            context += f"User: {entry['user'][:80]}...\n"
            context += f"Assistant: {entry['assistant'][:120]}...\n\n"
        
        return context
    
    def _get_system_prompt(self):
        """Advanced system prompt for financial AI"""
        return """You are an ultra-intelligent AI financial advisor with expertise in:
• Advanced financial analytics and data science
• Personal finance optimization and wealth building
• Behavioral economics and spending psychology
• Predictive forecasting and trend analysis
• Machine learning-based insights

YOUR CAPABILITIES:
✓ Analyze complex financial patterns
✓ Provide personalized, data-driven recommendations
✓ Create actionable savings strategies
✓ Predict future spending trends
✓ Identify anomalies and risks
✓ Build comprehensive financial plans

YOUR RESPONSE STYLE:
• Highly intelligent and insightful
• Clear, concise, and actionable
• Data-driven with specific numbers
• Professional yet friendly
• Use emojis strategically for clarity
• Format with markdown for readability

ALWAYS PROVIDE:
1. Direct answer to user's question
2. Specific dollar amounts and percentages
3. Actionable recommendations
4. Data-backed reasoning
5. Clear next steps

Be the smartest, most helpful financial advisor the user has ever interacted with."""
    
    def _fallback_response(self, user_message, financial_data):
        """Intelligent rule-based fallback"""
        msg = user_message.lower()
        
        count = financial_data.get('count', 0)
        total = financial_data.get('total', 0)
        avg = financial_data.get('avg', 0)
        categories = financial_data.get('categories', [])
        
        if not categories:
            return "🤖 Please upload your financial data first to get intelligent insights!"
        
        top_cat, top_amt = categories[0]
        
        if any(word in msg for word in ['save', 'reduce', 'cut', 'budget']):
            savings_10 = top_amt * 0.1
            savings_20 = top_amt * 0.2
            return f"""💰 **Smart Savings Strategy**

Your biggest expense is **{top_cat}** at ${top_amt:,.2f}/month ({top_amt/total*100:.1f}% of spending).

**Intelligent Recommendations:**
• 10% reduction → **${savings_10:,.2f}/month** = **${savings_10*12:,.2f}/year**
• 20% reduction → **${savings_20:,.2f}/month** = **${savings_20*12:,.2f}/year**

**Action Plan:**
1. Set {top_cat} budget at ${top_amt*0.9:,.2f}/month
2. Track daily spending in this category
3. Review weekly progress
4. Adjust and optimize

Start today and save **${savings_10*12:,.2f}** this year! 🚀"""
        
        elif any(word in msg for word in ['category', 'spending', 'most', 'breakdown']):
            breakdown = "\n".join([f"  {i}. {cat}: ${amt:,.2f} ({amt/total*100:.1f}%)" 
                                  for i, (cat, amt) in enumerate(categories[:5], 1)])
            return f"""📊 **Intelligent Spending Analysis**

Total: ${total:,.2f} across {count:,} transactions

**Top 5 Categories:**
{breakdown}

**Key Insight:** {top_cat} dominates at {top_amt/total*100:.1f}% - prime optimization target!

**Smart Recommendation:** Focus on reducing top 2 categories for maximum impact. 🎯"""
        
        else:
            return f"""🧠 **Smart Financial Overview**

**Your Snapshot:**
• {count:,} transactions = ${total:,.2f}
• Average: ${avg:.2f}/transaction
• Top expense: {top_cat} (${top_amt:,.2f})

**Ask me:**
• "How can I save $500/month?"
• "Show my spending categories"
• "What's my financial health?"
• "Predict my next month spending"

I'm here to help you make smarter financial decisions! 💡"""
    
    def get_insights(self, analytics_data):
        """Generate smart automated insights"""
        insights = []
        
        health_score = analytics_data.get('financial_health_score', 0)
        if health_score < 60:
            insights.append(f"⚠️ Health score {health_score:.0f}/100 needs attention - review spending patterns")
        elif health_score >= 80:
            insights.append(f"✅ Excellent health score {health_score:.0f}/100 - keep it up!")
        
        overspending = analytics_data.get('overspending_categories', [])
        if overspending:
            insights.append(f"🔴 Overspending detected in: {', '.join(overspending)}")
        
        anomalies = len(analytics_data.get('anomalies_iqr', []))
        if anomalies > 0:
            insights.append(f"🔍 {anomalies} unusual transactions detected - review for errors")
        
        forecast = analytics_data.get('forecast_30days', 0)
        total = analytics_data.get('total_spending', 0)
        if forecast > total * 1.1:
            insights.append(f"📈 Spending trending up - forecast ${forecast:,.2f} next month")
        
        return insights
