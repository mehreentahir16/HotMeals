"""
Agent tools for the HotMeals restaurant assistant.

These tools allow the LLM to interact with the restaurant database.
Compatible with LangChain 1.2+
"""

from langchain.tools import tool
import json

from src.database import (
    search_restaurants,
    get_restaurant_by_name,
    get_restaurant_by_id,
    is_open_now,
)

@tool
def search_restaurants_tool(query: str) -> str:
    """Search for restaurants based on criteria like cuisine, location, rating, and price.
    
    Args:
        query: JSON string with optional fields: cuisine, city, state, min_stars, max_price, limit
        
    Example: {"cuisine": "Italian", "city": "Philadelphia", "min_stars": 4.0}
    """
    try:
        params = json.loads(query)
        
        results = search_restaurants(
            cuisine=params.get('cuisine'),
            city=params.get('city'),
            state=params.get('state'),
            min_stars=params.get('min_stars'),
            max_price=params.get('max_price'),
            limit=params.get('limit', 10)
        )
        
        if not results:
            return "No restaurants found matching your criteria. Try broadening your search or check the city/state name."
        
        output = f"Found {len(results)} restaurant(s):\n\n"
        for i, restaurant in enumerate(results, 1):
            price_range = "N/A"
            if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
                price = restaurant['attributes'].get('RestaurantsPriceRange2')
                if price:
                    try:
                        price_range = '$' * int(price)
                    except:
                        pass
            
            output += f"{i}. **{restaurant['name']}**\n"
            output += f"   📍 {restaurant['address']}, {restaurant['city']}, {restaurant['state']}\n"
            output += f"   ⭐ {restaurant['stars']} stars ({restaurant['review_count']} reviews)\n"
            output += f"   💰 {price_range}\n"
            output += f"   🍽️  {restaurant['categories']}\n"
            output += f"   Status: {'🟢 Open' if restaurant['is_open'] == 1 else '🔴 Closed'}\n"
            output += f"   ID: {restaurant['business_id']}\n\n"
        
        return output
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid query format. Please provide a valid JSON string. Error: {str(e)}"
    except Exception as e:
        return f"Error searching restaurants: {str(e)}"

@tool
def get_restaurant_details_tool(query: str) -> str:
    """Get detailed information about a specific restaurant including address, hours, ratings, and amenities.
    
    Args:
        query: JSON string with either 'name' and optional 'city', or 'business_id'
        
    Example: {"name": "Vetri Cucina", "city": "Philadelphia"}
    """
    try:
        params = json.loads(query)
        
        if 'business_id' in params:
            restaurant = get_restaurant_by_id(params['business_id'])
        elif 'name' in params:
            restaurant = get_restaurant_by_name(params['name'], params.get('city'))
        else:
            return "Error: Please provide either 'name' or 'business_id'"
        
        if not restaurant:
            return "Restaurant not found. Please check the name or ID and try again."
        
        output = f"**{restaurant['name']}**\n\n"
        output += f"📍 Address:\n   {restaurant['address']}\n   {restaurant['city']}, {restaurant['state']} {restaurant['postal_code']}\n\n"
        output += f"⭐ Rating: {restaurant['stars']} stars ({restaurant['review_count']} reviews)\n\n"
        
        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            price = restaurant['attributes'].get('RestaurantsPriceRange2')
            if price:
                try:
                    output += f"💰 Price Range: {'$' * int(price)}\n\n"
                except:
                    pass
        
        output += f"🍽️  Categories: {restaurant['categories']}\n\n"
        output += f"Status: {'🟢 Currently Open' if restaurant['is_open'] == 1 else '🔴 Currently Closed'}\n\n"
        
        if restaurant.get('hours') and isinstance(restaurant['hours'], dict):
            output += "🕒 Hours:\n"
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                output += f"   {day}: {restaurant['hours'].get(day, 'N/A')}\n"
            output += "\n"
        
        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            attrs = restaurant['attributes']
            features = []
            
            if attrs.get('RestaurantsTakeOut') == 'True':
                features.append("✓ Takeout available")
            if attrs.get('RestaurantsDelivery') == 'True':
                features.append("✓ Delivery available")
            if attrs.get('OutdoorSeating') == 'True':
                features.append("✓ Outdoor seating")
            if attrs.get('WiFi') and 'free' in str(attrs.get('WiFi')).lower():
                features.append("✓ Free WiFi")
            if attrs.get('GoodForKids') == 'True':
                features.append("✓ Good for kids")
            if attrs.get('WheelchairAccessible') == 'True':
                features.append("✓ Wheelchair accessible")
            
            if features:
                output += "ℹ️  Features:\n"
                for feat in features:
                    output += f"   {feat}\n"
                output += "\n"
        
        output += f"🆔 Business ID: {restaurant['business_id']}\n"
        return output
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid query format. Error: {str(e)}"
    except Exception as e:
        return f"Error getting restaurant details: {str(e)}"

@tool
def check_if_open_tool(query: str) -> str:
    """Check if a restaurant is currently open based on their hours.
    
    Args:
        query: JSON string with either 'name' and optional 'city', or 'business_id'
        
    Example: {"name": "Vetri Cucina", "city": "Philadelphia"}
    """
    try:
        params = json.loads(query)
        
        if 'business_id' in params:
            restaurant = get_restaurant_by_id(params['business_id'])
        elif 'name' in params:
            restaurant = get_restaurant_by_name(params['name'], params.get('city'))
        else:
            return "Error: Please provide either 'name' or 'business_id'"
        
        if not restaurant:
            return "Restaurant not found. Please check the name or ID and try again."
        
        is_open, message = is_open_now(restaurant.get('hours'))
        
        output = f"**{restaurant['name']}** in {restaurant['city']}, {restaurant['state']}\n"
        output += f"✅ {message}" if is_open else f"❌ {message}"
        
        if restaurant.get('hours'):
            from datetime import datetime
            today = datetime.now().strftime("%A")
            if today_hours := restaurant['hours'].get(today):
                output += f"\nToday's hours: {today_hours}"
        
        return output
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid query format. Error: {str(e)}"
    except Exception as e:
        return f"Error checking restaurant hours: {str(e)}"

# Export all tools
all_tools = [
    search_restaurants_tool,
    get_restaurant_details_tool,
    check_if_open_tool
]