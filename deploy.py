import os
import requests
import json

from google import genai
from google.genai import types
from datetime import datetime
from pydantic import BaseModel


class EnergyPriceInfo(BaseModel):
    price: float
    last_update: datetime
    valid_from: datetime


# --- Configuration ---
PROVIDER_URL = os.environ.get("ENERGY_PROVIDER_URL")

client = genai.Client()


def fetch_and_parse_price():
    """
    Fetches the HTML from the provider's URL and uses the Gemini API
    to parse the electricity price and the month it is valid from.
    """
    try:
        # Fetch the HTML content from the provider's website
        response = requests.get(PROVIDER_URL, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        html_content = response.text

        # Use the Gemini API to parse the HTML
        prompt = f"""
        You are an expert at parsing messy HTML. Your task is to find the electricity price in €/kWh and the start validity period from the following HTML content.

        HTML_CONTENT:
        ---
        {html_content}
        ---
        """
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=EnergyPriceInfo
            ),
        )

        response_text = result.text.strip()

        print("Response from Gemini API:", response_text)

        data = json.loads(response_text)

        price = float(data["price"])
        valid_from = data["valid_from"]
        last_updated = datetime.now().isoformat()

        return {"price": price, "last_update": last_updated, "valid_from": valid_from}

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    price_data = fetch_and_parse_price()
    if price_data:
        # Create a directory for deployment if it doesn't exist
        if not os.path.exists("public"):
            os.makedirs("public")

        # Write the data to index.json
        with open("public/index.json", "w", encoding="utf-8") as f:
            json.dump(price_data, f, indent=4)

        print("Successfully fetched price and updated index.json")
    else:
        print("Failed to fetch price data.")
        exit(1)
