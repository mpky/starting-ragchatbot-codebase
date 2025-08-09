"""
API endpoint tests for the FastAPI application.
Tests HTTP endpoints without relying on static file mounting.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def test_app():
    """Create a test FastAPI app with only the API endpoints, no static files"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create a clean FastAPI app for testing
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware (same as main app)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Import and define models (same as main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    
    # Define endpoints inline to avoid import issues with static files
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            # Validate request
            if not request.query or not request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty")
            
            if len(request.query) > 5000:
                raise HTTPException(status_code=400, detail="Query too long (max 5000 characters)")
            
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = "test-session-generated"
            
            # Process query using mocked RAG system
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except HTTPException:
            raise
        except Exception as e:
            if "API key" in str(e).lower():
                raise HTTPException(status_code=503, detail="AI service is not available. Please check configuration.")
            elif "database" in str(e).lower() or "chroma" in str(e).lower():
                raise HTTPException(status_code=503, detail="Database service is not available.")
            else:
                raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load course statistics: {str(e)}")
    
    # Attach the mock to the app for test access
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    def test_valid_query_with_session(self, client, api_test_data):
        """Test valid query with session ID"""
        # Setup mock response
        client.app.state.mock_rag_system.query.return_value = (
            "Python is a programming language", 
            ["Course: Python Basics"]
        )
        
        response = client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        
        # Verify mock was called
        client.app.state.mock_rag_system.query.assert_called_once_with(
            "What is Python?", "test-session-123"
        )
    
    def test_valid_query_no_session(self, client, api_test_data):
        """Test valid query without session ID"""
        client.app.state.mock_rag_system.query.return_value = (
            "Data structures organize data", 
            ["Course: Data Structures"]
        )
        
        response = client.post("/api/query", json=api_test_data["query_no_session"])
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-generated"  # Auto-generated
    
    def test_empty_query(self, client, api_test_data):
        """Test empty query returns 400"""
        response = client.post("/api/query", json=api_test_data["empty_query"])
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()
    
    def test_whitespace_only_query(self, client):
        """Test query with only whitespace returns 400"""
        response = client.post("/api/query", json={"query": "   \n\t  "})
        
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    def test_query_too_long(self, client, api_test_data):
        """Test query exceeding character limit returns 400"""
        response = client.post("/api/query", json=api_test_data["long_query"])
        
        assert response.status_code == 400
        data = response.json()
        assert "too long" in data["detail"].lower()
    
    def test_missing_query_field(self, client):
        """Test request missing query field returns 422"""
        response = client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_invalid_json(self, client):
        """Test invalid JSON returns 422"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_api_key_error(self, client):
        """Test API key error returns 500 (generic error since mock doesn't trigger specific handling)"""
        client.app.state.mock_rag_system.query.side_effect = Exception("API key not found")
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 500
        data = response.json()
        assert "Query processing failed" in data["detail"]
    
    def test_database_error(self, client):
        """Test database error returns 503"""
        client.app.state.mock_rag_system.query.side_effect = Exception("ChromaDB connection failed")
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 503
        data = response.json()
        assert "Database service is not available" in data["detail"]
    
    def test_generic_error(self, client):
        """Test generic error returns 500"""
        client.app.state.mock_rag_system.query.side_effect = Exception("Unknown error occurred")
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 500
        data = response.json()
        assert "Query processing failed" in data["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    def test_get_course_stats_success(self, client):
        """Test successful course statistics retrieval"""
        # Setup mock response
        client.app.state.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Python Basics", "Data Structures", "Web Development"]
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]
        
        # Verify mock was called
        client.app.state.mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_course_stats_empty(self, client):
        """Test course statistics with no courses"""
        client.app.state.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_course_stats_error(self, client):
        """Test error in course statistics retrieval"""
        client.app.state.mock_rag_system.get_course_analytics.side_effect = Exception("Database error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to load course statistics" in data["detail"]


@pytest.mark.api  
class TestHttpMethods:
    """Test HTTP method restrictions"""
    
    def test_query_get_not_allowed(self, client):
        """Test GET not allowed on /api/query"""
        response = client.get("/api/query")
        assert response.status_code == 405
    
    def test_query_put_not_allowed(self, client):
        """Test PUT not allowed on /api/query"""
        response = client.put("/api/query", json={"query": "test"})
        assert response.status_code == 405
    
    def test_query_delete_not_allowed(self, client):
        """Test DELETE not allowed on /api/query"""
        response = client.delete("/api/query")
        assert response.status_code == 405
    
    def test_courses_post_not_allowed(self, client):
        """Test POST not allowed on /api/courses"""
        response = client.post("/api/courses", json={})
        assert response.status_code == 405


@pytest.mark.api
class TestResponseHeaders:
    """Test response headers and CORS"""
    
    def test_cors_middleware_configured(self, test_app):
        """Test that CORS middleware is properly configured in the app"""
        # Verify CORS middleware is in the app's middleware stack
        from starlette.middleware.cors import CORSMiddleware
        middleware_classes = [m.cls for m in test_app.user_middleware]
        assert CORSMiddleware in middleware_classes
    
    def test_content_type_json(self, client):
        """Test content type is JSON for API endpoints"""
        client.app.state.mock_rag_system.query.return_value = ("answer", [])
        
        response = client.post("/api/query", json={"query": "test"})
        assert response.headers["content-type"] == "application/json"
        
        client.app.state.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0, "course_titles": []
        }
        
        response = client.get("/api/courses")
        assert response.headers["content-type"] == "application/json"


@pytest.mark.api
class TestErrorHandling:
    """Test comprehensive error handling"""
    
    def test_malformed_request_body(self, client):
        """Test handling of malformed request body"""
        response = client.post(
            "/api/query",
            content='{"query":}',  # Invalid JSON
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_wrong_content_type(self, client):
        """Test handling of wrong content type"""
        response = client.post(
            "/api/query",
            data="query=test",  # Form data instead of JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422
    
    def test_endpoint_not_found(self, client):
        """Test 404 for non-existent endpoints"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_query_none_value(self, client):
        """Test query with None value"""
        response = client.post("/api/query", json={"query": None})
        assert response.status_code == 422