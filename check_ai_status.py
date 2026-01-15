"""Quick check of AI initialization"""
import os
from dotenv import load_dotenv
load_dotenv()

print("\n" + "="*60)
print("AI STATUS CHECK")
print("="*60)

# Check API key
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"\n1. API Key exists: {bool(api_key)}")
if api_key:
    print(f"   Starts with sk-ant-: {api_key.startswith('sk-ant-')}")
    print(f"   Length: {len(api_key)}")

# Try importing modules
try:
    from smart_ai_engine import UltraSmartAI
    print(f"\n2. smart_ai_engine imported: ✅")
    
    # Try initializing
    ultra_ai = UltraSmartAI(anthropic_key=api_key)
    print(f"3. UltraSmartAI created: ✅")
    print(f"4. Has anthropic_client: {ultra_ai.anthropic_client is not None}")
    
    if ultra_ai.anthropic_client:
        print(f"5. Model: {ultra_ai.claude_model}")
        print(f"\n{'='*60}")
        print("✅ AI IS READY")
        print("="*60)
    else:
        print(f"\n{'='*60}")
        print("❌ AI CLIENT NOT INITIALIZED")
        print("="*60)
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
