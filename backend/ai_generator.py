from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anthropic


@dataclass
class ConversationRound:
    """Represents a single round of conversation with context"""

    round_number: int
    messages: List[Dict[str, Any]]
    tools_used: List[str] = None
    completed: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search and outline tools for course information.

Tool Selection and Sequential Usage:
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews (returns course title, course link, and complete numbered lesson list)
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials  
- **Sequential Tool Usage**: You can make up to 2 tool calls in sequence per user query
  - Use the first tool call to gather initial information
  - If needed, make a second tool call to gather additional or more specific information
  - After 2 tool calls, provide your final response based on all gathered information
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Sequential Tool Strategy Examples:
- **Broad then Specific**: First search broadly, then search within specific course/lesson based on initial results
- **Multiple Courses**: Search one course, then search another if the first doesn't have relevant content
- **Outline then Content**: Get course outline first, then search specific lesson content
- **Content then Outline**: Search for content, then get course structure if more context needed

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_sequential_tool_execution(
                response, api_params, tool_manager, max_rounds=2
            )

        # Return direct response
        return response.content[0].text

    def _handle_sequential_tool_execution(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        max_rounds: int = 2,
    ) -> str:
        """
        Handle sequential tool execution with up to max_rounds of tool calling.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool execution rounds (default: 2)

        Returns:
            Final response text after sequential tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response

        for round_num in range(1, max_rounds + 1):
            # Check if current response has tool use
            if not self._has_tool_use(current_response):
                # No tool use - return text response
                return self._extract_text_response(current_response)

            # Add AI's response with tool use to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools and get results
            tool_results, error = self._execute_tools_for_response(
                current_response, tool_manager
            )

            if error:
                # Tool execution failed - return error handling
                return self._handle_tool_error(error, current_response)

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

            # Check if this is the last round
            if round_num == max_rounds:
                # Final round - call without tools to force text response
                final_response = self._make_api_call(
                    messages, base_params["system"], tools=None
                )
                return self._extract_text_response(final_response)
            else:
                # Continue with tools available for next round
                next_response = self._make_api_call(
                    messages, base_params["system"], base_params.get("tools")
                )
                current_response = next_response

        # Fallback - should not reach here
        return self._extract_text_response(current_response)

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy method - redirects to sequential handler for backward compatibility
        """
        return self._handle_sequential_tool_execution(
            initial_response, base_params, tool_manager, max_rounds=1
        )

    def _has_tool_use(self, response) -> bool:
        """Check if response contains tool use blocks"""
        return response.stop_reason == "tool_use" and any(
            content_block.type == "tool_use" for content_block in response.content
        )

    def _extract_text_response(self, response) -> str:
        """Extract text content from response"""
        for content_block in response.content:
            if content_block.type == "text":
                return content_block.text
        return "No text response available"

    def _execute_tools_for_response(
        self, response, tool_manager
    ) -> Tuple[List[Dict], Optional[str]]:
        """Execute all tool calls in response and return results with any error"""
        tool_results = []
        try:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
            return tool_results, None
        except Exception as e:
            return tool_results, str(e)

    def _make_api_call(
        self, messages: List[Dict], system_prompt: str, tools: Optional[List] = None
    ):
        """Make an API call to Claude with given parameters"""
        api_params = {**self.base_params, "messages": messages, "system": system_prompt}

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _handle_tool_error(self, error: str, last_response) -> str:
        """Handle tool execution errors gracefully"""
        # Try to return any text response from Claude if available
        text_response = self._extract_text_response(last_response)
        if text_response and text_response != "No text response available":
            return text_response

        # Return a helpful error message
        return f"I encountered an error while searching for information. Please try rephrasing your question."
