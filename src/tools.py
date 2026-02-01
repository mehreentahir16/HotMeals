"""
Agent tools for the BiteBot restaurant assistant.

These tools allow the LLM to interact with the restaurant database.
Compatible with LangChain 1.2+
"""

import uuid
import json
from typing import Optional
from typing import Dict, List
from datetime import datetime, timedelta

import logging
import re

from langchain.tools import tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.database import (
    search_restaurants,
    get_restaurant_by_name,
    get_restaurant_by_id,
    is_open_now,
)

@tool
def search_restaurants_tool(
    cuisine: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    min_stars: Optional[float] = None,
    max_price: Optional[int] = None,
    has_takeout: Optional[bool] = None,
    has_delivery: Optional[bool] = None,
    outdoor_seating: Optional[bool] = None,
    wheelchair_accessible: Optional[bool] = None,
    good_for_kids: Optional[bool] = None,
    has_wifi: Optional[bool] = None,
    accepts_reservations: Optional[bool] = None,
    good_for_groups: Optional[bool] = None,
    limit: int = 10
) -> str:
    """Search for restaurants based on criteria like cuisine, location, rating, price, and amenities.
    
    Args:
        query: JSON string with optional fields:
            - cuisine: cuisine type (e.g., 'Italian', 'Mexican', 'Chinese')
            - city: city name (e.g., 'Philadelphia', 'Tampa')
            - state: state abbreviation (e.g., 'PA', 'FL')
            - min_stars: minimum rating (1.0-5.0)
            - max_price: max price range (1-4)
            - has_takeout: true/false for takeout availability
            - has_delivery: true/false for delivery availability
            - outdoor_seating: true/false for outdoor seating
            - wheelchair_accessible: true/false for accessibility
            - good_for_kids: true/false for kid-friendly
            - has_wifi: true/false for WiFi availability
            - accepts_reservations: true/false for reservation capability
            - good_for_groups: true/false for group-friendly
            - limit: max results (default 10)
        
    Example: {"cuisine": "Italian", "city": "Philadelphia", "min_stars": 4.0, "outdoor_seating": true, "accepts_reservations": true}
    """
    try:
        results = search_restaurants(
            cuisine=cuisine,
            city=city,
            state=state,
            min_stars=min_stars,
            max_price=max_price,
            has_takeout=has_takeout,
            has_delivery=has_delivery,
            outdoor_seating=outdoor_seating,
            wheelchair_accessible=wheelchair_accessible,
            good_for_kids=good_for_kids,
            has_wifi=has_wifi,
            accepts_reservations=accepts_reservations,
            good_for_groups=good_for_groups,
            limit=limit
        )

        logger.info(f"[SEARCH] Found {len(results)} results")
        
        if not results:
            return "No restaurants found matching your criteria. Try broadening your search."
        
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
        
    except Exception as e:
        logger.error(f"[SEARCH] Error: {e}", exc_info=True)
        return f"Error searching restaurants: {str(e)}"

@tool
def get_restaurant_details_tool(
    name: Optional[str] = None,
    city: Optional[str] = None,
    business_id: Optional[str] = None
) -> str:
    """Get detailed information about a specific restaurant.
    
    Args:
        name: Restaurant name
        city: City name (optional, for disambiguation)
        business_id: Yelp business ID (alternative to name)
    """
    logger.info(f"[DETAILS] Called with name={name}, city={city}, business_id={business_id}")
    
    try:
        if business_id:
            restaurant = get_restaurant_by_id(business_id)
        elif name:
            restaurant = get_restaurant_by_name(name, city)
        else:
            return "Error: Please provide either 'name' or 'business_id'"
        
        if not restaurant:
            return "Restaurant not found. Please check the name and try again."
        
        logger.info(f"[DETAILS] Found: {restaurant['name']}")
        
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
        
        # Pass ALL attributes as JSON
        if restaurant.get('attributes') and isinstance(restaurant['attributes'], dict):
            output += "ℹ️  Amenities & Features:\n"
            output += json.dumps(restaurant['attributes'], indent=2)
            output += "\n\n"
        
        output += f"🆔 Business ID: {restaurant['business_id']}\n"
        return output
        
    except Exception as e:
        logger.error(f"[DETAILS] Error: {e}", exc_info=True)
        return f"Error getting restaurant details: {str(e)}"

@tool
def check_availability_tool(
    name: Optional[str] = None,
    city: Optional[str] = None,
    business_id: Optional[str] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    party_size: int = 2
) -> str:
    """Check if a restaurant is open now OR check table availability.
    
    Args:
        name: Restaurant name
        city: City name (optional)
        business_id: Yelp business ID (alternative to name)
        date: Date - can be "today", "tomorrow", "next thursday", or YYYY-MM-DD
        time: Time - can be "7pm", "19:00", etc.
        party_size: Number of people (default 2)
    """
    logger.info(f"[CHECK_AVAILABILITY] Called with name={name}, date={date}, time={time}, party_size={party_size}")
    
    try:
        if business_id:
            restaurant = get_restaurant_by_id(business_id)
        elif name:
            restaurant = get_restaurant_by_name(name, city)
        else:
            return "Error: Please provide either 'name' or 'business_id'"
        
        if not restaurant:
            return "Restaurant not found."
        
        logger.info(f"[CHECK_AVAILABILITY] Found: {restaurant['name']}")
        
        now = datetime.now()
        
        # If time provided, check table availability
        if time or date:
            # Parse date
            if date:
                date_lower = date.lower()
                if date_lower == 'today':
                    reservation_date = now
                elif date_lower == 'tomorrow':
                    reservation_date = now + timedelta(days=1)
                elif 'next' in date_lower:
                    days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    target_day = None
                    for day in days_of_week:
                        if day in date_lower:
                            target_day = days_of_week.index(day)
                            break
                    
                    if target_day is not None:
                        current_day = now.weekday()
                        days_ahead = (target_day - current_day + 7) % 7
                        if days_ahead == 0:
                            days_ahead = 7
                        reservation_date = now + timedelta(days=days_ahead)
                    else:
                        reservation_date = datetime.strptime(date, '%Y-%m-%d')
                else:
                    reservation_date = datetime.strptime(date, '%Y-%m-%d')
            else:
                reservation_date = now
            
            logger.info(f"[CHECK_AVAILABILITY] Parsed date: {reservation_date.strftime('%Y-%m-%d')}")
            
            # Parse time
            if time:
                time_lower = time.lower()
                if 'pm' in time_lower or 'am' in time_lower:
                    match = re.match(r'(\d+):?(\d*)\s*(am|pm)', time_lower)
                    if match:
                        hour = int(match.group(1))
                        minute = int(match.group(2)) if match.group(2) else 0
                        period = match.group(3)
                        
                        if period == 'pm' and hour != 12:
                            hour += 12
                        elif period == 'am' and hour == 12:
                            hour = 0
                        
                        reservation_time = f"{hour:02d}:{minute:02d}"
                    else:
                        reservation_time = time
                else:
                    reservation_time = time
            else:
                reservation_time = now.strftime('%H:%M')
            
            logger.info(f"[CHECK_AVAILABILITY] Parsed time: {reservation_time}")
            
            # Check hours
            day_name = reservation_date.strftime("%A")
            if restaurant.get('hours') and day_name in restaurant['hours']:
                day_hours = restaurant['hours'][day_name]
                
                req_hour, req_min = map(int, reservation_time.split(':'))
                req_minutes = req_hour * 60 + req_min
                
                open_time, close_time = day_hours.split('-')
                open_hour, open_min = map(int, open_time.split(':'))
                close_hour, close_min = map(int, close_time.split(':'))
                open_minutes = open_hour * 60 + open_min
                close_minutes = close_hour * 60 + close_min
                
                if open_minutes == 0 and close_minutes == 0:
                    return f"❌ **{restaurant['name']}** is closed on {day_name}."
                
                if close_minutes < open_minutes:
                    time_ok = req_minutes >= open_minutes or req_minutes < close_minutes
                else:
                    time_ok = open_minutes <= req_minutes < close_minutes
                
                if not time_ok:
                    return f"❌ **{restaurant['name']}** is not open at {reservation_time} on {day_name}.\nHours: {day_hours}"
                
                date_str = reservation_date.strftime("%Y-%m-%d")
                output = f"✅ **{restaurant['name']}**\n"
                output += f"📅 {date_str} ({day_name}) at {reservation_time}\n"
                output += f"👥 Party of {party_size}\n\n"
                output += f"Table available! Restaurant is open from {open_time} to {close_time}.\n"
                output += f"📍 {restaurant['address']}, {restaurant['city']}, {restaurant['state']}\n"
                
                logger.info(f"[CHECK_AVAILABILITY] Success!")
                return output
            else:
                return f"Hours not available for {day_name}."
        
        # Check if open now
        is_open, message = is_open_now(restaurant.get('hours'))
        
        output = f"**{restaurant['name']}** in {restaurant['city']}, {restaurant['state']}\n"
        output += f"✅ {message}" if is_open else f"❌ {message}"
        
        if restaurant.get('hours'):
            today = now.strftime("%A")
            if today_hours := restaurant['hours'].get(today):
                output += f"\nToday's hours: {today_hours}"
        
        return output
        
    except Exception as e:
        logger.error(f"[CHECK_AVAILABILITY] Error: {e}", exc_info=True)
        return f"Error: {str(e)}"

@tool
def make_reservation_tool(
    name: Optional[str] = None,
    city: Optional[str] = None,
    business_id: Optional[str] = None,
    date: str = "",  # Must be YYYY-MM-DD format
    time: str = "",  # Must be HH:MM format (24-hour)
    party_size: int = 2,
    customer_name: str = "",
    customer_phone: Optional[str] = None,
    special_requests: Optional[str] = None
) -> str:
    """Make a reservation at a restaurant.
    
    Args:
        name: Restaurant name
        city: City name (optional)
        business_id: Yelp business ID
        date: Reservation date in YYYY-MM-DD format (get this from check_availability_tool response)
        time: Reservation time in HH:MM 24-hour format (get this from check_availability_tool response)
        party_size: Number of people
        customer_name: Name for reservation
        customer_phone: Contact phone
        special_requests: Any special requests
    """
    logger.info(f"[MAKE_RESERVATION] Called for {name}, date={date}, time={time}")
    
    try:
        if business_id:
            restaurant = get_restaurant_by_id(business_id)
        elif name:
            restaurant = get_restaurant_by_name(name, city)
        else:
            return "Error: Please provide restaurant name"
        
        if not restaurant:
            return "Restaurant not found."
        
        if not all([date, time, customer_name]):
            return "Error: Missing required fields (date, time, customer_name)"
        
        # Validate date format (must be YYYY-MM-DD)
        try:
            reservation_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return f"Error: Date must be in YYYY-MM-DD format. You provided: {date}. Please check availability first to get the correct date format."
        
        # Validate time format (must be HH:MM)
        try:
            datetime.strptime(time, '%H:%M')
        except ValueError:
            return f"Error: Time must be in HH:MM 24-hour format. You provided: {time}. Please check availability first to get the correct time format."
        
        now = datetime.now()
        
        # Check if date is in the past
        if reservation_date.date() < now.date():
            return "Error: Cannot make reservations for past dates."
        
        import uuid
        reservation_id = str(uuid.uuid4())[:8]
        
        reservation = {
            'reservation_id': reservation_id,
            'restaurant_name': restaurant['name'],
            'restaurant_id': restaurant['business_id'],
            'address': f"{restaurant['address']}, {restaurant['city']}, {restaurant['state']}",
            'date': date,
            'time': time,
            'party_size': party_size,
            'customer_name': customer_name,
            'customer_phone': customer_phone or 'Not provided',
            'special_requests': special_requests or 'None',
            'status': 'confirmed',
            'created_at': now.isoformat()
        }
        
        output = f"🎉 **Reservation Confirmed!**\n\n"
        output += f"📋 Confirmation #: {reservation_id}\n\n"
        output += f"🍽️  **{restaurant['name']}**\n"
        output += f"📍 {reservation['address']}\n\n"
        output += f"📅 Date: {date}\n"
        output += f"🕐 Time: {time}\n"
        output += f"👥 Party Size: {party_size}\n"
        output += f"👤 Name: {customer_name}\n"
        output += f"📞 Phone: {reservation['customer_phone']}\n"
        
        if special_requests:
            output += f"💬 Special Requests: {special_requests}\n"
        
        output += f"\n✅ Your table is reserved!\n"
        output += f"📝 Confirmation: {reservation_id}\n\n"
        output += f"IMPORTANT: This reservation data includes: {json.dumps(reservation)}"
        
        logger.info(f"[MAKE_RESERVATION] Success! ID: {reservation_id}")
        return output
        
    except Exception as e:
        logger.error(f"[MAKE_RESERVATION] Error: {e}", exc_info=True)
        return f"Error: {str(e)}"

# Export all tools
all_tools = [
    search_restaurants_tool,
    get_restaurant_details_tool,
    check_availability_tool,
    make_reservation_tool
]
