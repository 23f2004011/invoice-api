from fastapi import FastAPI, Request
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1",
)

@app.post("/extract")
async def extract(request: Request):
    body = await request.json()
    text = body["text"]
    schema = body["schema"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured invoice data from messy text. "
                    "Follow these rules exactly:\n"
                    "- currency: convert to ISO 4217 code (USD, EUR, GBP, INR, JPY)\n"
                    "- total_amount: plain integer, no separators, no symbols, "
                    "convert spelled-out numbers, K/M suffixes, and Indian-style "
                    "grouping (1,24,800) to a normal integer\n"
                    "- invoice_date: YYYY-MM-DD\n"
                    "- due_in_days: integer, convert phrases like 'Net 30' or "
                    "'two weeks' to a day count\n"
                    "- is_paid: boolean inferred from wording\n"
                    "- priority: one of low, normal, high, urgent\n"
                    "- contact_email: lowercase\n"
                    "- line_items: array in the order they appear, unit_price "
                    "as integer\n"
                    "- item_count: number of line items\n"
                    "Return ONLY the JSON object, matching the schema exactly. "
                    "No extra keys, no missing keys, no commentary."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice",
                "schema": schema,
                "strict": True,
            },
        },
    )

    import json
    return json.loads(response.choices[0].message.content)