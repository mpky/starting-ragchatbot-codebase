import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the backend directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_successful_search_with_results(self):
        """Test successful search that returns results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is lesson content about Python", "More Python content"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = self.search_tool.execute("Python basics")
        
        # Verify the vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert "[Python Basics - Lesson 1]" in result
        assert "[Python Basics - Lesson 2]" in result
        assert "This is lesson content about Python" in result
        assert "More Python content" in result
        
        # Verify sources were tracked
        assert len(self.search_tool.last_sources) == 2
        assert self.search_tool.last_sources[0]["text"] == "Python Basics - Lesson 1"
        assert self.search_tool.last_sources[0]["url"] == "https://example.com/lesson1"
    
    def test_search_with_course_name_filter(self):
        """Test search with course name filtering"""
        mock_results = SearchResults(
            documents=["Course specific content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("functions", course_name="Advanced Python")
        
        # Verify the vector store was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        assert "[Advanced Python - Lesson 3]" in result
        assert "Course specific content" in result
    
    def test_search_with_lesson_number_filter(self):
        """Test search with lesson number filtering"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson5"
        
        result = self.search_tool.execute("statistics", lesson_number=5)
        
        # Verify the vector store was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="statistics",
            course_name=None,
            lesson_number=5
        )
        
        assert "[Data Science - Lesson 5]" in result
        assert "Lesson specific content" in result
    
    def test_search_with_both_filters(self):
        """Test search with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Very specific content"],
            metadata=[{"course_title": "Machine Learning", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/ml-lesson2"
        
        result = self.search_tool.execute("neural networks", course_name="Machine Learning", lesson_number=2)
        
        # Verify the vector store was called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="Machine Learning",
            lesson_number=2
        )
        
        assert "[Machine Learning - Lesson 2]" in result
        assert "Very specific content" in result
    
    def test_search_with_no_results(self):
        """Test search that returns no results"""
        # Mock empty search results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        assert result == "No relevant content found."
        assert len(self.search_tool.last_sources) == 0
    
    def test_search_with_no_results_and_course_filter(self):
        """Test search with no results and course filter"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic", course_name="Nonexistent Course")
        
        assert result == "No relevant content found in course 'Nonexistent Course'."
    
    def test_search_with_no_results_and_lesson_filter(self):
        """Test search with no results and lesson filter"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic", lesson_number=99)
        
        assert result == "No relevant content found in lesson 99."
    
    def test_search_with_no_results_and_both_filters(self):
        """Test search with no results and both filters"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("topic", course_name="Course", lesson_number=1)
        
        assert result == "No relevant content found in course 'Course' in lesson 1."
    
    def test_search_with_vector_store_error(self):
        """Test search when vector store returns an error"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("any query")
        
        assert result == "Database connection failed"
        assert len(self.search_tool.last_sources) == 0
    
    def test_search_results_formatting_without_lesson_number(self):
        """Test formatting of results without lesson numbers"""
        mock_results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "Overview Course", "lesson_number": None}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("overview")
        
        assert "[Overview Course]" in result
        assert "General course content" in result
        
        # Verify source tracking for content without lesson numbers
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0]["text"] == "Overview Course"
        assert self.search_tool.last_sources[0]["url"] is None
    
    def test_search_results_with_missing_metadata(self):
        """Test handling of results with missing metadata fields"""
        mock_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("test")
        
        assert "[unknown]" in result
        assert "Content with missing metadata" in result
    
    def test_multiple_search_calls_reset_sources(self):
        """Test that sources are properly tracked across multiple calls"""
        # First search
        mock_results1 = SearchResults(
            documents=["First result"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results1
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        self.search_tool.execute("first query")
        first_sources = self.search_tool.last_sources.copy()
        
        # Second search
        mock_results2 = SearchResults(
            documents=["Second result"],
            metadata=[{"course_title": "Course 2", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results2
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"
        
        self.search_tool.execute("second query")
        second_sources = self.search_tool.last_sources
        
        # Verify sources are replaced, not accumulated
        assert len(first_sources) == 1
        assert len(second_sources) == 1
        assert first_sources != second_sources
        assert second_sources[0]["text"] == "Course 2 - Lesson 2"


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__])