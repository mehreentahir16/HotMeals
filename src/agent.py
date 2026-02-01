"""
LangChain agent setup for BiteBot restaurant assistant.

This module creates the agent that orchestrates the LLM, tools, and memory.
Compatible with LangChain 1.2+ and OpenAI 2.15+
"""

import os
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.tools import all_tools

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_bitebot_agent():
    """
    Create and return the BiteBot restaurant agent.
    
    Returns:
        Agent: The configured LangChain agent
    """
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError(
            "OpenAI API key not found. "
            "Please create a .env file with your OPENAI_API_KEY"
        )
    
    # Initialize the LLM
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
    )
    
    # System prompt for the agent
    system_prompt = """You are BiteBot, a friendly and helpful restaurant assistant. 

        PERSONALITY:
        - Warm, enthusiastic, and helpful
        - Present information naturally in paragraphs where possible

        TOOLS:
        - search_restaurants_tool: Find restaurants
        - get_restaurant_details_tool: Get full info
        - check_availability_tool: Check hours OR table availability (accepts natural language dates/times)
        - make_reservation_tool: Book tables (requires exact YYYY-MM-DD and HH:MM format)

        IMPORTANT - RESERVATION WORKFLOW:
        1. When user wants to book, ALWAYS call check_availability_tool FIRST with natural language
        2. Extract the exact date and time from the availability response (it will be formatted as YYYY-MM-DD and HH:MM)
        3. Use those exact values when calling make_reservation_tool

        Example:
        User: "Book tomorrow at 7pm for 4"
        1. Call: check_availability_tool(date="tomorrow", time="7pm", party_size=4)
        2. Response: "✅ Table available! 2026-02-02 (Sunday) at 19:00..."
        3. Extract: date=2026-02-02, time=19:00
        4. Call: make_reservation_tool(date="2026-02-02", time="19:00", party_size=4, customer_name=...)

        Be conversational and helpful!"""
    
    logger.info(f"Creating agent with {len(all_tools)} tools")
    for tool in all_tools:
        logger.info(f"  - Tool: {tool.name}")
    
    # Create the agent with the new API
    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt
    )
    
    return agent

def run_agent(agent, user_input: str, conversation_history: list = None):
    """
    Run the agent with user input and conversation history.
    
    Args:
        agent: The Agent instance
        user_input: The user's message
        conversation_history: List of previous messages in the conversation
        
    Returns:
        dict: Response from the agent
    """
    try:
        # Build the full message history
        messages = []
        
        # Add previous conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add the new user message
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        logger.info(f"[AGENT] Invoking agent with {len(messages)} messages")
        logger.info(f"[AGENT] Latest user message: {user_input}")
        
        # Invoke the agent with full conversation history
        response = agent.invoke({"messages": messages})
        
        logger.info(f"[AGENT] Response received")
        logger.info(f"[AGENT] Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        
        # Extract the final message
        if response and "messages" in response:
            response_messages = response["messages"]
            logger.info(f"[AGENT] Got {len(response_messages)} response messages")
            
            # LOG ALL MESSAGES TO SEE TOOL CALLS
            for i, msg in enumerate(response_messages):
                msg_type = type(msg).__name__
                logger.info(f"[AGENT] Message {i}: {msg_type}")
                if hasattr(msg, 'content'):
                    logger.info(f"[AGENT]   Content preview: {str(msg.content)[:100]}")
                if hasattr(msg, 'tool_calls'):
                    logger.info(f"[AGENT]   Tool calls: {msg.tool_calls}")
                if msg_type == 'ToolMessage':
                    logger.info(f"[AGENT]   Tool result: {str(msg)[:200]}")
            
            if response_messages:
                last_message = response_messages[-1]
                logger.info(f"[AGENT] Last message type: {type(last_message)}")
                
                output = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                return {
                    "output": output,
                    "intermediate_steps": []
                }
        
        logger.warning("[AGENT] No messages in response")
        return {
            "output": "I apologize, but I couldn't process that request.",
            "intermediate_steps": []
        }
    except Exception as e:
        logger.error(f"[AGENT] Error: {e}", exc_info=True)
        return {
            "output": f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your request.",
            "intermediate_steps": []
        }