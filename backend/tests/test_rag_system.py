import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
from dataclasses import dataclass

# Add the backend directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test_api_key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


class TestRAGSystem:
    """Test suite for RAGSystem end-to-end functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        self.mock_config = MockConfig()
        
        # Mock all dependencies
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            self.rag_system = RAGSystem(self.mock_config)
            
            # Set up mocks for testing
            self.mock_ai_generator = Mock()
            self.mock_tool_manager = Mock()
            self.mock_session_manager = Mock()
            self.mock_vector_store = Mock()
            
            self.rag_system.ai_generator = self.mock_ai_generator
            self.rag_system.tool_manager = self.mock_tool_manager
            self.rag_system.session_manager = self.mock_session_manager
            self.rag_system.vector_store = self.mock_vector_store
    
    def test_query_success_with_api_key(self):
        """Test successful query processing when API key is available"""
        # Set up mocks for successful flow
        self.mock_ai_generator.generate_response.return_value = "AI response about Python"
        self.mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        self.mock_tool_manager.get_last_sources.return_value = [{"text": "Python Course - Lesson 1", "url": "http://example.com"}]
        self.mock_tool_manager.reset_sources.return_value = None
        self.mock_session_manager.get_conversation_history.return_value = None
        self.mock_session_manager.add_exchange.return_value = None
        
        result, sources = self.rag_system.query("What is Python?", session_id="test_session")
        
        # Verify API key was checked (should pass)
        assert result == "AI response about Python"
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Course - Lesson 1"
        
        # Verify all components were called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        self.mock_tool_manager.get_tool_definitions.assert_called_once()
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
        self.mock_session_manager.add_exchange.assert_called_once_with("test_session", "What is Python?", "AI response about Python")
    
    def test_query_no_api_key(self):
        """Test query handling when no API key is provided"""
        # Remove API key
        self.rag_system.config.ANTHROPIC_API_KEY = ""
        
        result, sources = self.rag_system.query("What is Python?")
        
        # Should return placeholder message without calling AI
        assert result == "I will tell you the answer once I am plugged in (have an API key)."
        assert sources == []
        
        # Verify AI generator was not called
        self.mock_ai_generator.generate_response.assert_not_called()
    
    def test_query_no_api_key_none(self):
        """Test query handling when API key is None"""
        # Set API key to None
        self.rag_system.config.ANTHROPIC_API_KEY = None
        
        result, sources = self.rag_system.query("What is Python?")
        
        assert result == "I will tell you the answer once I am plugged in (have an API key)."
        assert sources == []
        self.mock_ai_generator.generate_response.assert_not_called()
    
    def test_query_with_session_history(self):
        """Test query processing with conversation history"""
        # Set up session history
        self.mock_session_manager.get_conversation_history.return_value = "Previous: Hello\nAssistant: Hi there!"
        self.mock_ai_generator.generate_response.return_value = "Follow up response"
        self.mock_tool_manager.get_tool_definitions.return_value = []
        self.mock_tool_manager.get_last_sources.return_value = []
        
        result, sources = self.rag_system.query("Follow up question", session_id="session_123")
        
        # Verify history was retrieved and used
        self.mock_session_manager.get_conversation_history.assert_called_once_with("session_123")
        
        # Verify AI generator was called with history
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous: Hello\nAssistant: Hi there!"
        
        assert result == "Follow up response"
    
    def test_query_without_session(self):
        """Test query processing without session ID"""
        self.mock_ai_generator.generate_response.return_value = "Response without session"
        self.mock_tool_manager.get_tool_definitions.return_value = []
        self.mock_tool_manager.get_last_sources.return_value = []
        
        result, sources = self.rag_system.query("Test query")
        
        # Verify session manager was not called for history
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator was called with no history
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] is None
        
        assert result == "Response without session"
    
    def test_query_ai_generator_exception(self):
        """Test handling of AI generator exceptions"""
        # Simulate AI generator throwing an exception
        self.mock_ai_generator.generate_response.side_effect = Exception("API call failed")
        self.mock_tool_manager.get_tool_definitions.return_value = []
        
        # This should raise the exception (not caught in the current implementation)
        with pytest.raises(Exception) as exc_info:
            self.rag_system.query("Test query")
        
        assert str(exc_info.value) == "API call failed"
    
    def test_query_tool_integration(self):
        """Test integration between query and tool system"""
        # Set up tool definitions
        mock_tool_defs = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = mock_tool_defs
        
        # Set up sources from tools
        mock_sources = [
            {"text": "Python Basics - Lesson 1", "url": "https://example.com/lesson1"},
            {"text": "Python Basics - Lesson 2", "url": "https://example.com/lesson2"}
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        self.mock_ai_generator.generate_response.return_value = "AI response using tools"
        
        result, sources = self.rag_system.query("Tell me about Python")
        
        # Verify tool definitions were passed to AI
        ai_call_args = self.mock_ai_generator.generate_response.call_args
        assert ai_call_args[1]["tools"] == mock_tool_defs
        assert ai_call_args[1]["tool_manager"] == self.mock_tool_manager
        
        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
        
        assert result == "AI response using tools"
        assert sources == mock_sources
    
    def test_query_prompt_format(self):
        """Test that query is properly formatted for AI"""
        self.mock_ai_generator.generate_response.return_value = "Test response"
        self.mock_tool_manager.get_tool_definitions.return_value = []
        self.mock_tool_manager.get_last_sources.return_value = []
        
        user_query = "What is machine learning?"
        self.rag_system.query(user_query)
        
        # Verify the prompt format
        ai_call_args = self.mock_ai_generator.generate_response.call_args
        expected_prompt = f"Answer this question about course materials: {user_query}"
        # The query parameter is passed as a keyword argument
        assert ai_call_args.kwargs["query"] == expected_prompt
    
    def test_add_course_document_success(self):
        """Test successful course document processing"""
        # Mock document processor
        mock_course = Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1")
            ]
        )
        mock_chunks = [
            CourseChunk(content="Test content", course_title="Test Course", lesson_number=1, chunk_index=0)
        ]
        
        self.rag_system.document_processor = Mock()
        self.rag_system.document_processor.process_course_document.return_value = (mock_course, mock_chunks)
        
        course, chunk_count = self.rag_system.add_course_document("test_file.txt")
        
        # Verify processing and storage
        self.rag_system.document_processor.process_course_document.assert_called_once_with("test_file.txt")
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)
        
        assert course == mock_course
        assert chunk_count == 1
    
    def test_add_course_document_failure(self):
        """Test handling of course document processing failure"""
        # Mock document processor to raise exception
        self.rag_system.document_processor = Mock()
        self.rag_system.document_processor.process_course_document.side_effect = Exception("File not found")
        
        course, chunk_count = self.rag_system.add_course_document("nonexistent_file.txt")
        
        # Should handle exception gracefully
        assert course is None
        assert chunk_count == 0
        
        # Vector store should not be called
        self.mock_vector_store.add_course_metadata.assert_not_called()
        self.mock_vector_store.add_course_content.assert_not_called()
    
    def test_get_course_analytics(self):
        """Test course analytics retrieval"""
        # Mock vector store analytics
        self.mock_vector_store.get_course_count.return_value = 3
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3"
        ]
        
        analytics = self.rag_system.get_course_analytics()
        
        expected = {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
        
        assert analytics == expected
        self.mock_vector_store.get_course_count.assert_called_once()
        self.mock_vector_store.get_existing_course_titles.assert_called_once()
    
    def test_query_source_cleanup(self):
        """Test that sources are properly cleaned up after each query"""
        # First query
        self.mock_tool_manager.get_last_sources.return_value = [{"text": "Source 1"}]
        self.mock_ai_generator.generate_response.return_value = "Response 1"
        self.mock_tool_manager.get_tool_definitions.return_value = []
        
        result1, sources1 = self.rag_system.query("Query 1")
        
        # Verify reset was called
        self.mock_tool_manager.reset_sources.assert_called()
        
        # Second query
        self.mock_tool_manager.get_last_sources.return_value = [{"text": "Source 2"}]
        self.mock_ai_generator.generate_response.return_value = "Response 2"
        
        result2, sources2 = self.rag_system.query("Query 2")
        
        # Verify reset was called again
        assert self.mock_tool_manager.reset_sources.call_count == 2
        
        # Verify sources are different
        assert sources1 != sources2


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__])