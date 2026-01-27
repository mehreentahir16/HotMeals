"""
LangChain agent setup for HotMeals restaurant assistant.

This module creates the agent that orchestrates the LLM, tools, and memory.
Compatible with LangChain 1.2+ and OpenAI 2.15+
"""

import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.tools import all_tools

# Load environment variables
load_dotenv()

def create_hotmeals_agent():
    """
    Create and return the HotMeals restaurant agent.
    
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
        model="gpt-4o-mini",  # Fast and affordable
        temperature=0.7,  # Slightly creative but focused
    )
    
    # System prompt for the agent
    system_prompt = """You are HotMeals, a friendly and helpful restaurant recommendation assistant.

Your role is to help users find restaurants that match their preferences using the tools available to you. You have access to a database of real restaurants from the Yelp dataset.

Guidelines:
- Be conversational and friendly
- Ask clarifying questions when needed (e.g., which city, what type of cuisine, price range)
- Use search_restaurants_tool to find restaurants matching criteria
- Use get_restaurant_details_tool to get comprehensive information about a specific restaurant
- Use check_if_open_tool to verify if a restaurant is currently open
- Provide recommendations based on ratings, reviews, and user preferences
- If a search returns no results, suggest alternatives or ask the user to modify their criteria
- Remember the conversation context to provide personalized recommendations

When using tools:
- Always provide input as valid JSON strings
- For searching, use: {"cuisine": "Italian", "city": "Philadelphia", "min_stars": 4.0}
- For details/open status, use: {"name": "Restaurant Name", "city": "City Name"}

Major cities in the dataset include: Philadelphia, Tampa, Phoenix, Indianapolis, Nashville, Tucson, New Orleans, and many more.

Remember: You're here to help users discover great restaurants and have an enjoyable dining experience!"""
    
    # Create the agent with the new API
    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt
    )
    
    return agent

def run_agent(agent, user_input: str):
    """
    Run the agent with user input.
    
    Args:
        agent: The Agent instance
        user_input: The user's message
        
    Returns:
        dict: Response from the agent
    """
    try:
        # Invoke the agent with the user input
        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        
        # Extract the final message
        if response and "messages" in response:
            messages = response["messages"]
            if messages:
                last_message = messages[-1]
                return {
                    "output": last_message.get("content", ""),
                    "intermediate_steps": []  # New API doesn't expose this the same way
                }
        
        return {
            "output": "I apologize, but I couldn't process that request.",
            "intermediate_steps": []
        }
    except Exception as e:
        return {
            "output": f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your request.",
            "intermediate_steps": []
        }