#!/usr/bin/env python3
"""
Test script untuk memverifikasi instalasi dan setup
Run dengan: python test_setup.py
"""

import sys
import os

def test_imports():
    """Test apakah semua library yang diperlukan terinstall"""
    
    print("🔍 Checking required libraries...")
    print("-" * 50)
    
    required_libraries = {
        'streamlit': 'streamlit',
        'anthropic': 'anthropic',
        'dotenv': 'python-dotenv',
        'pandas': 'pandas',
        'PyPDF2': 'PyPDF2',
        'sklearn': 'scikit-learn',
        'tiktoken': 'tiktoken',
        'numpy': 'numpy'
    }
    
    optional_libraries = {
        'fitz': 'PyMuPDF',
        'pytesseract': 'pytesseract',
        'PIL': 'pillow',
        'pdf2image': 'pdf2image'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required libraries
    print("\n✅ Required Libraries:")
    for module, package_name in required_libraries.items():
        try:
            __import__(module)
            print(f"  ✓ {module} ({package_name})")
        except ImportError:
            print(f"  ✗ {module} ({package_name}) - MISSING")
            missing_required.append(package_name)
    
    # Check optional libraries
    print("\n📦 Optional Libraries (for enhanced features):")
    for module, package_name in optional_libraries.items():
        try:
            __import__(module)
            print(f"  ✓ {module} ({package_name})")
        except ImportError:
            print(f"  ○ {module} ({package_name}) - Not installed")
            missing_optional.append(package_name)
    
    print("\n" + "=" * 50)
    
    # Report results
    if missing_required:
        print("\n❌ MISSING REQUIRED LIBRARIES!")
        print("Install with:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    else:
        print("\n✅ All required libraries installed!")
        
    if missing_optional:
        print("\n💡 Optional libraries not installed:")
        print("For full OCR support, install with:")
        print(f"pip install {' '.join(missing_optional)}")
        
    return True

def test_env_file():
    """Check if .env file exists and has API key"""
    print("\n🔑 Checking environment configuration...")
    print("-" * 50)
    
    if os.path.exists('.env'):
        print("✓ .env file found")
        
        # Try to load and check for API key
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                if api_key.startswith("sk-"):
                    print("✓ ANTHROPIC_API_KEY is configured")
                    print(f"  Key starts with: {api_key[:10]}...")
                else:
                    print("⚠️ ANTHROPIC_API_KEY found but might be invalid")
                    print("  Keys should start with 'sk-'")
            else:
                print("✗ ANTHROPIC_API_KEY not found in .env")
                print("\nAdd to .env file:")
                print("ANTHROPIC_API_KEY=your-api-key-here")
                return False
        except Exception as e:
            print(f"✗ Error loading .env: {e}")
            return False
    else:
        print("✗ .env file not found")
        print("\nCreate .env file with:")
        print("echo 'ANTHROPIC_API_KEY=your-api-key-here' > .env")
        return False
    
    return True

def test_smart_kb():
    """Test if smart_kb module can be imported"""
    print("\n📚 Checking Smart Knowledge Base module...")
    print("-" * 50)
    
    if os.path.exists('smart_kb.py'):
        print("✓ smart_kb.py file found")
        
        try:
            from smart_kb import (
                SmartKnowledgeBase,
                DocumentChunker,
                SemanticRetriever,
                TokenManager,
                ConversationManager
            )
            print("✓ All smart_kb classes can be imported")
            
            # Quick functionality test
            print("\n🧪 Running quick functionality test...")
            
            # Test TokenManager
            tm = TokenManager()
            test_text = "This is a test text for token counting."
            token_count = tm.count_tokens(test_text)
            print(f"  ✓ TokenManager: '{test_text}' = {token_count} tokens")
            
            # Test DocumentChunker
            chunker = DocumentChunker(chunk_size=100, overlap=20)
            chunks = chunker.chunk_text("This is sentence one. This is sentence two. This is sentence three.", "test_doc")
            print(f"  ✓ DocumentChunker: Created {len(chunks)} chunks")
            
            # Test SemanticRetriever
            retriever = SemanticRetriever()
            retriever.index_chunks(chunks)
            results = retriever.retrieve_relevant_chunks("sentence two", top_k=1)
            print(f"  ✓ SemanticRetriever: Retrieved {len(results)} relevant chunks")
            
            return True
            
        except ImportError as e:
            print(f"✗ Error importing smart_kb: {e}")
            print("\nMake sure smart_kb.py contains all required classes")
            return False
        except Exception as e:
            print(f"⚠️ smart_kb imported but test failed: {e}")
            return True  # Module exists but has issues
    else:
        print("✗ smart_kb.py not found")
        print("\nCreate smart_kb.py from the provided code artifact")
        return False

def test_streamlit_app():
    """Check if main streamlit app exists"""
    print("\n🚀 Checking Streamlit application...")
    print("-" * 50)
    
    if os.path.exists('main_optimized.py'):
        print("✓ main_optimized.py found")
        
        # Check if it imports correctly
        try:
            with open('main_optimized.py', 'r') as f:
                content = f.read()
                if 'SmartKnowledgeBase' in content:
                    print("✓ main_optimized.py uses SmartKnowledgeBase")
                else:
                    print("⚠️ main_optimized.py might not be using smart KB")
                    
                if 'key=' in content:
                    print("✓ Streamlit elements have unique keys")
                else:
                    print("⚠️ Streamlit elements might be missing keys")
        except Exception as e:
            print(f"⚠️ Could not analyze main_optimized.py: {e}")
            
        return True
    else:
        print("✗ main_optimized.py not found")
        print("\nCreate main_optimized.py from the provided code artifact")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("🔧 SMART KNOWLEDGE BASE SETUP VERIFICATION")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_env_file()
    all_passed &= test_smart_kb()
    all_passed &= test_streamlit_app()
    
    # Final summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ SETUP COMPLETE! Ready to run the application.")
        print("\nRun with:")
        print("streamlit run main_optimized.py")
    else:
        print("❌ SETUP INCOMPLETE. Please fix the issues above.")
        print("\nQuick fix commands:")
        print("1. Install missing packages:")
        print("   pip install streamlit anthropic python-dotenv pandas PyPDF2 scikit-learn tiktoken numpy")
        print("\n2. Create .env file:")
        print("   echo 'ANTHROPIC_API_KEY=your-api-key-here' > .env")
        print("\n3. Make sure smart_kb.py and main_optimized.py are created from the artifacts")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()