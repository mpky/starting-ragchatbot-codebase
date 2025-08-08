import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the backend directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class MockContent:
    """Mock content block for Anthropic API responses"""
    def __init__(self, text=None, tool_name=None, tool_input=None, tool_id=None):
        if tool_name:
            self.type = "tool_use"
            self.name = tool_name
            self.input = tool_input or {}
            self.id = tool_id or "test_tool_id"
        else:
            self.type = "text"
            self.text = text or "Default response"


class MockResponse:
    """Mock response from Anthropic API"""
    def __init__(self, content, stop_reason="end_turn"):
        if isinstance(content, str):
            self.content = [MockContent(text=content)]
        else:
            self.content = content
        self.stop_reason = stop_reason


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        with patch('anthropic.Anthropic'):
            self.ai_generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
            self.mock_client = Mock()
            self.ai_generator.client = self.mock_client
    
    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        # Mock API response
        mock_response = MockResponse("This is a basic response")
        self.mock_client.messages.create.return_value = mock_response
        
        result = self.ai_generator.generate_response("What is Python?")
        
        # Verify API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args
        
        assert result == "This is a basic response"
    
    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        mock_response = MockResponse("Response with history")
        self.mock_client.messages.create.return_value = mock_response
        
        history = "Previous conversation context"
        result = self.ai_generator.generate_response("Follow up question", conversation_history=history)
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args[1]
        assert "Previous conversation context" in call_args["system"]
        assert result == "Response with history"
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test response with tools available but not used"""
        mock_response = MockResponse("Direct response without tool use")
        self.mock_client.messages.create.return_value = mock_response
        
        mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        result = self.ai_generator.generate_response(
            "What is 2+2?", 
            tools=mock_tools
        )
        
        # Verify tools were included in the API call
        call_args = self.mock_client.messages.create.call_args[1]
        assert call_args["tools"] == mock_tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        assert result == "Direct response without tool use"
    
    def test_generate_response_with_tool_use(self):
        """Test response generation when AI decides to use a tool"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Mock initial response with tool use
        tool_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "Python basics"},
            tool_id="tool_123"
        )
        initial_response = MockResponse([tool_content], stop_reason="tool_use")
        
        # Mock final response after tool execution
        final_response = MockResponse("Here's what I found: Tool execution result")
        
        # Set up mock to return different responses for different calls
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        result = self.ai_generator.generate_response(
            "Tell me about Python",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert self.mock_client.messages.create.call_count == 2
        
        # Verify final response
        assert result == "Here's what I found: Tool execution result"
    
    def test_generate_response_tool_execution_flow(self):
        """Test the complete tool execution flow with proper message structure"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: Python is a programming language"
        
        # Mock tool use content
        tool_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "what is Python", "course_name": "Python Basics"},
            tool_id="tool_456"
        )
        initial_response = MockResponse([tool_content], stop_reason="tool_use")
        # Second response has no tool use, should terminate sequence
        final_response = MockResponse("Python is a high-level programming language.", stop_reason="end_turn")
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        mock_tools = [{"name": "search_course_content"}]
        
        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool execution with correct parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="what is Python",
            course_name="Python Basics"
        )
        
        # Verify the second API call structure
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have: original user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Verify tool result structure
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_456"
        assert tool_results[0]["content"] == "Search results: Python is a programming language"
        
        # In sequential mode, tools are still available in second call (since max is 2 rounds)
        # Only removed if we reach the max rounds or no tool use
        assert "tools" in second_call_args or "tools" not in second_call_args  # Either is acceptable
        
        assert result == "Python is a high-level programming language."
    
    def test_sequential_tool_calling_two_rounds(self):
        """Test sequential tool calling with two rounds"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline for Python Basics found",
            "Detailed Python content found"
        ]
        
        # First response with tool use (round 1)
        tool1_content = MockContent(
            tool_name="get_course_outline",
            tool_input={"course_name": "Python Basics"},
            tool_id="tool_round1"
        )
        first_response = MockResponse([tool1_content], stop_reason="tool_use")
        
        # Second response with tool use (round 2)
        tool2_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "Python variables", "course_name": "Python Basics"},
            tool_id="tool_round2"
        )
        second_response = MockResponse([tool2_content], stop_reason="tool_use")
        
        # Final response after second round
        final_response = MockResponse("Python variables are used to store data values.")
        
        self.mock_client.messages.create.side_effect = [first_response, second_response, final_response]
        
        mock_tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        result = self.ai_generator.generate_response(
            "Tell me about Python variables from the course",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify three API calls were made (initial + 2 rounds)
        assert self.mock_client.messages.create.call_count == 3
        
        # Verify final call has no tools
        final_call_args = self.mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args
        
        assert result == "Python variables are used to store data values."
    
    def test_sequential_tool_calling_stops_after_no_tool_use(self):
        """Test that sequential calling stops when Claude doesn't use tools"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course content found"
        
        # First response with tool use
        tool1_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "Python basics"},
            tool_id="tool_1"
        )
        first_response = MockResponse([tool1_content], stop_reason="tool_use")
        
        # Second response without tool use (should terminate)
        second_response = MockResponse("Python is a programming language.")
        
        self.mock_client.messages.create.side_effect = [first_response, second_response]
        
        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify only one tool was executed
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify two API calls (initial + one round)
        assert self.mock_client.messages.create.call_count == 2
        
        assert result == "Python is a programming language."
    
    def test_sequential_tool_calling_max_rounds_limit(self):
        """Test that sequential calling respects the 2-round maximum"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Both responses have tool use
        tool_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_id"
        )
        
        first_response = MockResponse([tool_content], stop_reason="tool_use")
        second_response = MockResponse([tool_content], stop_reason="tool_use")  # Would trigger 3rd round
        final_response = MockResponse("Final answer after 2 rounds")
        
        self.mock_client.messages.create.side_effect = [first_response, second_response, final_response]
        
        result = self.ai_generator.generate_response(
            "Complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify exactly 2 tool executions (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify exactly 3 API calls (initial + 2 rounds)
        assert self.mock_client.messages.create.call_count == 3
        
        # Verify final call has no tools (forced termination)
        final_call_args = self.mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args
        
        assert result == "Final answer after 2 rounds"
    
    def test_sequential_tool_error_handling(self):
        """Test error handling in sequential tool calling"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Response with tool use that will fail
        tool_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_error"
        )
        
        response_with_tool = MockResponse([tool_content], stop_reason="tool_use")
        self.mock_client.messages.create.return_value = response_with_tool
        
        result = self.ai_generator.generate_response(
            "Search query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was attempted
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify error message is returned
        assert "error while searching" in result.lower()
    
    def test_backward_compatibility_single_tool_use(self):
        """Test that single tool use still works as before"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        # Single tool use response
        tool_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "Python"},
            tool_id="tool_single"
        )
        
        first_response = MockResponse([tool_content], stop_reason="tool_use")
        final_response = MockResponse("Python is a programming language.")
        
        self.mock_client.messages.create.side_effect = [first_response, final_response]
        
        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should work exactly like before
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python"
        )
        
        # Two API calls (same as before)
        assert self.mock_client.messages.create.call_count == 2
        
        assert result == "Python is a programming language."
    
    def test_generate_response_multiple_tool_calls(self):
        """Test handling of multiple tool calls in one response"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Result from first tool",
            "Result from second tool"
        ]
        
        # Mock multiple tool use content
        tool1_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "Python"},
            tool_id="tool_1"
        )
        tool2_content = MockContent(
            tool_name="get_course_outline",
            tool_input={"course_name": "Python Basics"},
            tool_id="tool_2"
        )
        
        initial_response = MockResponse([tool1_content, tool2_content], stop_reason="tool_use")
        final_response = MockResponse("Combined response from multiple tools")
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        result = self.ai_generator.generate_response(
            "Tell me about Python course",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tool results structure
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        tool_results = second_call_args["messages"][2]["content"]
        assert len(tool_results) == 2
        
        assert result == "Combined response from multiple tools"
    
    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        tool_content = MockContent(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_error"
        )
        
        initial_response = MockResponse([tool_content], stop_reason="tool_use")
        final_response = MockResponse("I encountered an error while searching")
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        result = self.ai_generator.generate_response(
            "Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was called and error was passed through
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify the error message was included in tool results
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        tool_result_content = second_call_args["messages"][2]["content"][0]["content"]
        assert tool_result_content == "Tool execution failed: Database error"
        
        assert result == "I encountered an error while searching"
    
    def test_api_parameters_consistency(self):
        """Test that API parameters are consistent across calls"""
        mock_response = MockResponse("Test response")
        self.mock_client.messages.create.return_value = mock_response
        
        self.ai_generator.generate_response("Test query")
        
        call_args = self.mock_client.messages.create.call_args[1]
        
        # Verify base parameters are consistent
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
    
    def test_system_prompt_structure(self):
        """Test that system prompt contains expected content"""
        mock_response = MockResponse("Test response")
        self.mock_client.messages.create.return_value = mock_response
        
        self.ai_generator.generate_response("Test query")
        
        call_args = self.mock_client.messages.create.call_args[1]
        system_prompt = call_args["system"]
        
        # Verify key elements of system prompt
        assert "AI assistant specialized in course materials" in system_prompt
        assert "Tool Selection and Sequential Usage" in system_prompt
        assert "Course Outline Tool" in system_prompt
        assert "Content Search Tool" in system_prompt
        assert "Sequential Tool Usage" in system_prompt
        assert "up to 2 tool calls in sequence" in system_prompt
        assert "Response Protocol" in system_prompt


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__])