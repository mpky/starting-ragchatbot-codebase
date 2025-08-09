"""
Shared test fixtures and configuration for the RAG system tests.
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Generator

# Add the backend directory to the Python path so we can import our modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from vector_store import VectorStore
from search_tools import CourseSearchTool, ToolManager


@dataclass
class TestConfig:
    """Test configuration for RAG system components"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    CHUNK_SIZE: int = 200  # Smaller for testing
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RESULTS: int = 3
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create and cleanup temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir: str) -> TestConfig:
    """Test configuration with temporary database path"""
    config = TestConfig()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")
    return config


@pytest.fixture
def test_config_no_api_key(temp_dir: str) -> TestConfig:
    """Test configuration without API key for testing failure modes"""
    config = TestConfig()
    config.ANTHROPIC_API_KEY = ""
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")
    return config


@pytest.fixture
def sample_course_content() -> str:
    """Sample course content for testing"""
    return """Course Title: Test Python Course
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


@pytest.fixture
def sample_course_file(temp_dir: str, sample_course_content: str) -> str:
    """Create a temporary course file with sample content"""
    course_file = os.path.join(temp_dir, "test_course.txt")
    with open(course_file, 'w') as f:
        f.write(sample_course_content)
    return course_file


@pytest.fixture
def vector_store(test_config: TestConfig) -> VectorStore:
    """Initialize vector store with test configuration"""
    return VectorStore(
        test_config.CHROMA_PATH,
        test_config.EMBEDDING_MODEL,
        test_config.MAX_RESULTS
    )


@pytest.fixture
def rag_system(test_config: TestConfig) -> RAGSystem:
    """Initialize RAG system with test configuration"""
    return RAGSystem(test_config)


@pytest.fixture
def rag_system_no_api_key(test_config_no_api_key: TestConfig) -> RAGSystem:
    """Initialize RAG system without API key"""
    return RAGSystem(test_config_no_api_key)


@pytest.fixture
def loaded_rag_system(rag_system: RAGSystem, sample_course_file: str) -> RAGSystem:
    """RAG system with sample course loaded"""
    rag_system.add_course_document(sample_course_file)
    return rag_system


@pytest.fixture
def search_tool(vector_store: VectorStore) -> CourseSearchTool:
    """Initialize course search tool"""
    return CourseSearchTool(vector_store)


@pytest.fixture
def tool_manager() -> ToolManager:
    """Initialize tool manager"""
    return ToolManager()


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls"""
    with patch('anthropic.Anthropic') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock the messages.create method
        mock_response = Mock()
        mock_response.content = [Mock(text="Mocked AI response")]
        mock_instance.messages.create.return_value = mock_response
        
        yield mock_instance


@pytest.fixture
def mock_vector_store():
    """Mock vector store for isolated testing"""
    mock_store = Mock()
    
    # Mock search results
    from vector_store import SearchResult
    mock_store.search.return_value = SearchResult(
        chunks=[],
        sources=[],
        error=None
    )
    
    return mock_store


@pytest.fixture
def api_test_data():
    """Sample data for API endpoint testing"""
    return {
        "valid_query": {
            "query": "What is Python?",
            "session_id": "test-session-123"
        },
        "query_no_session": {
            "query": "Explain data structures"
        },
        "empty_query": {
            "query": "",
            "session_id": "test-session-123"
        },
        "long_query": {
            "query": "x" * 6000,  # Exceeds 5000 char limit
            "session_id": "test-session-123"
        }
    }