#!/usr/bin/env python3
"""
Test Gemini API connection and basic functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test basic Gemini API connection."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        print("🔑 Testing Gemini API Connection")
        print("=" * 40)
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
        
        print(f"✅ API Key loaded: {api_key[:10]}...")
        
        # Initialize the model
        print("🚀 Initializing Gemini model...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_retries=2,
            api_key=api_key,
        )
        
        # Test a simple query
        print("💬 Testing simple query...")
        test_question = "What is 2+2?"
        response = llm.invoke(test_question)
        
        print(f"Question: {test_question}")
        print(f"Response: {response.content}")
        
        if "4" in response.content:
            print("✅ Gemini API is working correctly!")
            return True
        else:
            print("⚠️ Unexpected response from Gemini API")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

def test_direct_answer_node():
    """Test the direct answer functionality."""
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from agent.graph import direct_answer
        from langchain_core.messages import HumanMessage
        
        print("\n🎯 Testing Direct Answer Node")
        print("=" * 40)
        
        # Create test state
        state = {
            "messages": [HumanMessage(content="What is the capital of France?")],
            "reasoning_model": "gemini-2.0-flash"
        }
        
        # Create minimal config
        config = {
            "configurable": {}
        }
        
        print("📤 Sending test question to direct_answer node...")
        print(f"Question: {state['messages'][0].content}")
        
        # Call the direct answer function
        result = direct_answer(state, config)
        
        print("📥 Response received:")
        if 'messages' in result and result['messages']:
            answer = result['messages'][0].content
            print(f"Answer: {answer[:200]}...")
            
            # Check if it mentions Paris (expected answer)
            if "Paris" in answer:
                print("✅ Direct answer node working correctly!")
                return True
            else:
                print("⚠️ Unexpected answer content")
                return False
        else:
            print("❌ No response message found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing direct answer node: {e}")
        return False

def main():
    """Run connection tests."""
    print("🧪 Gemini Connection & Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("Gemini API Connection", test_gemini_api),
        ("Direct Answer Node", test_direct_answer_node),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CONNECTION TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All connection tests passed!")
        print("✅ Gemini API is properly configured")
        print("✅ Direct answer functionality works")
        print("\nReady to test the full web search toggle!")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)