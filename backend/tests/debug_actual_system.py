#!/usr/bin/env python3
"""
Debug script to test the actual system and reproduce the "query failed" issue
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager


def test_vector_store_directly():
    """Test the vector store directly to see if it has data"""
    print("=== Testing Vector Store Directly ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Check if we have any courses
        course_count = vector_store.get_course_count()
        print(f"Number of courses in database: {course_count}")
        
        course_titles = vector_store.get_existing_course_titles()
        print(f"Course titles: {course_titles}")
        
        if course_count > 0:
            # Try a basic search
            print("\n--- Testing basic search ---")
            results = vector_store.search("Python programming")
            print(f"Search results error: {results.error}")
            print(f"Number of documents found: {len(results.documents)}")
            if results.documents:
                print(f"First document preview: {results.documents[0][:100]}...")
                print(f"Metadata: {results.metadata[0] if results.metadata else 'No metadata'}")
        else:
            print("No courses found - database might be empty")
            
    except Exception as e:
        print(f"Vector store test failed: {e}")
        import traceback
        traceback.print_exc()


def test_search_tool_directly():
    """Test the CourseSearchTool directly"""
    print("\n=== Testing CourseSearchTool Directly ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)
        
        # Test basic search
        print("--- Testing basic search: 'Python programming' ---")
        result = search_tool.execute("Python programming")
        print(f"Result: {result}")
        
        # Test search with course filter
        print("\n--- Testing search with course filter ---")
        result = search_tool.execute("Python", course_name="Building")
        print(f"Result with course filter: {result}")
        
        # Test search for something that definitely doesn't exist
        print("\n--- Testing search for nonexistent content ---")
        result = search_tool.execute("quantum computing advanced topics")
        print(f"Result for nonexistent content: {result}")
        
    except Exception as e:
        print(f"CourseSearchTool test failed: {e}")
        import traceback
        traceback.print_exc()


def test_tool_manager():
    """Test the ToolManager"""
    print("\n=== Testing ToolManager ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test tool definitions
        print("--- Tool definitions ---")
        tool_defs = tool_manager.get_tool_definitions()
        print(f"Number of tools: {len(tool_defs)}")
        if tool_defs:
            print(f"First tool definition: {tool_defs[0]}")
        
        # Test tool execution
        print("\n--- Testing tool execution ---")
        result = tool_manager.execute_tool("search_course_content", query="Python basics")
        print(f"Tool execution result: {result}")
        
        # Test getting sources
        sources = tool_manager.get_last_sources()
        print(f"Sources: {sources}")
        
    except Exception as e:
        print(f"ToolManager test failed: {e}")
        import traceback
        traceback.print_exc()


def test_rag_system():
    """Test the RAG system"""
    print("\n=== Testing RAG System ===")
    
    try:
        rag_system = RAGSystem(config)
        
        # Test analytics first
        print("--- Course analytics ---")
        analytics = rag_system.get_course_analytics()
        print(f"Analytics: {analytics}")
        
        # Test query without API key
        print("\n--- Testing query without API key ---")
        # Temporarily remove API key to test
        original_key = rag_system.config.ANTHROPIC_API_KEY
        rag_system.config.ANTHROPIC_API_KEY = ""
        
        result, sources = rag_system.query("What is Python?")
        print(f"Result without API key: {result}")
        print(f"Sources: {sources}")
        
        # Restore API key
        rag_system.config.ANTHROPIC_API_KEY = original_key
        
        # If we have an API key, test with it (but only if explicitly set)
        if original_key and len(original_key.strip()) > 0:
            print("\n--- Testing query with API key ---")
            try:
                result, sources = rag_system.query("What is Python programming?")
                print(f"Result with API key: {result[:200]}...")
                print(f"Number of sources: {len(sources)}")
                if sources:
                    print(f"First source: {sources[0]}")
            except Exception as e:
                print(f"Query with API key failed: {e}")
        else:
            print("\n--- No API key available for testing ---")
        
    except Exception as e:
        print(f"RAG system test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all debug tests"""
    print("Starting debug tests for the RAG chatbot system...")
    print(f"Using ChromaDB path: {config.CHROMA_PATH}")
    print(f"API key configured: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
    
    test_vector_store_directly()
    test_search_tool_directly()
    test_tool_manager()
    test_rag_system()
    
    print("\n=== Debug tests completed ===")


if __name__ == "__main__":
    main()