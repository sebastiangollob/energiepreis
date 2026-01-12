import os
import json
import time
import requests
from google import genai
from google.genai import types
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PROVIDER_URL = os.environ.get("ENERGY_PROVIDER_URL")
MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5

client = genai.Client(api_key=GOOGLE_API_KEY)


class EnergyPriceInfo(BaseModel):
    """
    Schema for electricity price data with built-in validation.
    """

    price: float = Field(
        description="The electricity price in €/kWh. Convert Cents to Euro (e.g., 14.5ct -> 0.145)."
    )
    valid_from: datetime = Field(description="The start date of the price validity.")

    @field_validator("price")
    @classmethod
    def price_must_be_realistic(cls, v: float) -> float:
        if v >= 1.0:
            raise ValueError(
                f"Extracted price {v} is too high (>1.00€). Likely a Cent-to-Euro conversion error."
            )
        if v <= 0:
            raise ValueError(f"Extracted price {v} must be positive.")
        return v


def fetch_and_parse_price() -> EnergyPriceInfo | None:
    if not PROVIDER_URL:
        raise ValueError("ENERGY_PROVIDER_URL environment variable is not set.")

    response = requests.get(PROVIDER_URL, timeout=15)
    response.raise_for_status()
    html_content = response.text

    prompt = f"""
    Extract the electricity price (€/kWh) and validity date from the HTML.
    
    CRITICAL CONVERSION RULES:
    1. If the price is in Cents (e.g., '35 ct' or '35 Cent'), you MUST divide by 100 to get Euro.
    2. A price of '14' is WRONG. It must be '0.14'.
    3. The price per kWh is almost always between 0.05 and 0.40 Euro. 
    4. If your result is >= 1.0, you have made a unit error. Correct it.

    HTML_CONTENT:
    ---
    {html_content}
    ---
    """

    retry_delay = INITIAL_RETRY_DELAY
    result = None

    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EnergyPriceInfo,
                    temperature=0.0,
                ),
            )
            break
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"Model overloaded (503). Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            raise e

    if not result or not result.text:
        raise ValueError("No response text received from the model")

    return EnergyPriceInfo.model_validate_json(result.text)


def main():
    try:
        price_data = fetch_and_parse_price()
    except Exception as e:
        print(f"Error fetching or parsing price data: {e}")
        exit(1)

    if price_data:
        os.makedirs("public", exist_ok=True)
        with open("public/index.json", "w", encoding="utf-8") as f:
            json.dump(price_data.model_dump(mode="json"), f, indent=4)
        print(f"Success: {price_data.price} €/kWh")
    else:
        print("Failed to update price due to validation error.")
        exit(1)


if __name__ == "__main__":
    main()
