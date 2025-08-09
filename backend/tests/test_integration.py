import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

import pytest

# Add the backend directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore


@dataclass
class TestConfig:
    """Test configuration with no API key to test the failure mode"""

    ANTHROPIC_API_KEY: str = ""  # No API key to test failure mode
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    CHUNK_SIZE: int = 200  # Smaller for testing
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RESULTS: int = 3
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


class TestIntegration:
    """Integration tests using real components to identify actual issues"""

    def setup_method(self):
        """Set up test fixtures with real components"""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = TestConfig()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")

        # Create test course content
        self.test_course_content = """Course Title: Test Python Course
Course Link: https://example.com/python-course
Course Instructor: Test Instructor

Lesson 1: Python Basics
Lesson Link: https://example.com/lesson1
Python is a high-level programming language. It's easy to learn and powerful for data science.
Variables in Python are created by assignment. You can store numbers, strings, and other data types.

Lesson 2: Data Structures
Lesson Link: https://example.com/lesson2
Python has several built-in data structures. Lists store ordered collections of items.
Dictionaries store key-value pairs and are very useful for data organization.
"""

        # Create test course file
        self.test_course_file = os.path.join(self.temp_dir, "test_course.txt")
        with open(self.test_course_file, "w") as f:
            f.write(self.test_course_content)

    def teardown_method(self):
        """Clean up test files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_vector_store_initialization(self):
        """Test that vector store can be initialized properly"""
        try:
            vector_store = VectorStore(
                self.test_config.CHROMA_PATH,
                self.test_config.EMBEDDING_MODEL,
                self.test_config.MAX_RESULTS,
            )

            # This should not raise an exception
            assert vector_store is not None
            assert hasattr(vector_store, "course_catalog")
            assert hasattr(vector_store, "course_content")

        except Exception as e:
            pytest.fail(f"Vector store initialization failed: {e}")

    def test_course_loading_and_search(self):
        """Test loading course documents and searching them"""
        try:
            # Initialize RAG system with test config
            rag_system = RAGSystem(self.test_config)

            # Add the test course
            course, chunk_count = rag_system.add_course_document(self.test_course_file)

            # Verify course was loaded
            assert course is not None
            assert course.title == "Test Python Course"
            assert course.instructor == "Test Instructor"
            assert len(course.lessons) == 2
            assert chunk_count > 0

            # Test direct search tool functionality
            search_tool = CourseSearchTool(rag_system.vector_store)

            # Test basic search
            result = search_tool.execute("Python programming")
            assert "No relevant content found" not in result
            assert "Python" in result or "programming" in result

            # Test course-specific search
            result = search_tool.execute("variables", course_name="Test Python Course")
            assert "No relevant content found" not in result

            # Test lesson-specific search
            result = search_tool.execute("data structures", lesson_number=2)
            assert "No relevant content found" not in result

        except Exception as e:
            pytest.fail(f"Course loading and search failed: {e}")

    def test_rag_system_query_without_api_key(self):
        """Test RAG system query when API key is missing"""
        try:
            # Initialize RAG system with no API key
            rag_system = RAGSystem(self.test_config)

            # Add test course
            rag_system.add_course_document(self.test_course_file)

            # Try to query - should return placeholder message
            result, sources = rag_system.query("What is Python?")

            expected_message = (
                "I will tell you the answer once I am plugged in (have an API key)."
            )
            assert result == expected_message
            assert sources == []

        except Exception as e:
            pytest.fail(f"RAG system query without API key failed: {e}")

    def test_tool_manager_functionality(self):
        """Test tool manager with real tools"""
        try:
            # Initialize components
            vector_store = VectorStore(
                self.test_config.CHROMA_PATH,
                self.test_config.EMBEDDING_MODEL,
                self.test_config.MAX_RESULTS,
            )

            # Load test data
            rag_system = RAGSystem(self.test_config)
            rag_system.add_course_document(self.test_course_file)

            # Test tool manager
            tool_manager = ToolManager()
            search_tool = CourseSearchTool(rag_system.vector_store)
            tool_manager.register_tool(search_tool)

            # Test tool definitions
            tool_defs = tool_manager.get_tool_definitions()
            assert len(tool_defs) == 1
            assert tool_defs[0]["name"] == "search_course_content"

            # Test tool execution
            result = tool_manager.execute_tool(
                "search_course_content", query="Python basics"
            )
            assert isinstance(result, str)
            assert "No relevant content found" not in result

        except Exception as e:
            pytest.fail(f"Tool manager functionality failed: {e}")

    def test_empty_database_search(self):
        """Test search behavior with empty database"""
        try:
            # Initialize with empty database
            vector_store = VectorStore(
                self.test_config.CHROMA_PATH,
                self.test_config.EMBEDDING_MODEL,
                self.test_config.MAX_RESULTS,
            )

            search_tool = CourseSearchTool(vector_store)

            # Search should return "no content found"
            result = search_tool.execute("anything")
            assert "No relevant content found" in result

        except Exception as e:
            pytest.fail(f"Empty database search failed: {e}")

    def test_search_with_nonexistent_course(self):
        """Test search with course name that doesn't exist"""
        try:
            rag_system = RAGSystem(self.test_config)
            rag_system.add_course_document(self.test_course_file)

            search_tool = CourseSearchTool(rag_system.vector_store)

            # Search for non-existent course
            result = search_tool.execute("Python", course_name="Nonexistent Course")
            assert (
                "No course found matching" in result
                or "No relevant content found" in result
            )

        except Exception as e:
            pytest.fail(f"Search with nonexistent course failed: {e}")

    def test_vector_store_error_conditions(self):
        """Test vector store error handling"""
        try:
            # Try to create vector store with invalid path
            invalid_path = "/invalid/path/that/doesnt/exist"

            # This might raise an exception or handle it gracefully
            try:
                vector_store = VectorStore(
                    invalid_path, self.test_config.EMBEDDING_MODEL, 5
                )
                # If it doesn't raise an exception, test search
                search_result = vector_store.search("test query")
                # Check if error is handled in search results
                if search_result.error:
                    assert "error" in search_result.error.lower()
            except Exception as e:
                # Exception is expected for invalid path
                assert (
                    "error" in str(e).lower()
                    or "permission" in str(e).lower()
                    or "path" in str(e).lower()
                )

        except Exception as e:
            pytest.fail(f"Vector store error condition test failed: {e}")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable",
    )
    def test_full_integration_with_api_key(self):
        """Test full integration when API key is available"""
        # Only run if we have an actual API key
        test_config_with_key = TestConfig()
        test_config_with_key.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
        test_config_with_key.CHROMA_PATH = os.path.join(
            self.temp_dir, "test_chroma_with_key"
        )

        try:
            rag_system = RAGSystem(test_config_with_key)
            rag_system.add_course_document(self.test_course_file)

            # This would make a real API call
            result, sources = rag_system.query("What is Python?")

            # Should get a real response, not the placeholder
            assert (
                result
                != "I will tell you the answer once I am plugged in (have an API key)."
            )
            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.fail(f"Full integration with API key failed: {e}")


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__])
