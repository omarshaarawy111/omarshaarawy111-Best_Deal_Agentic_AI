# Imports
import os
from dotenv import load_dotenv
from openai import OpenAI
import pickle
import time
from pathlib import Path
from tqdm import tqdm
# Class Item to validate the returned data and has lots of helper functions
from pricer.items import Item
import json
import truststore


load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
# Connect to OpenAI client library
# Create new intance of the OpenAI object
openai = OpenAI()

# Configurations
GENERATOR_MODEL = "gpt-4.1-nano"
# Input folder to LLM
BATCHES_FOLDER = "batches"
# Output folder from LLM
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")


SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

class Batch:
    BATCH_SIZE = 5_000
    batches = []

    # Intialization
    def __init__(self, items, start, end):
        # Item class object
        self.items = items
        # Configurations
        self.start = start
        self.end = end
        folder = Path("best_deal_batches") 
        self.filename = f"{start}_{end}.jsonl"
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        # State
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.done = False
        # Create directory if not exist
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    # Helper Functions
    def make_jsonl(self, item):
        body = {"model": GENERATOR_MODEL, 
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, 
                             {"role": "user", "content": item.full}
                             ]
                }
        line = {"custom_id": str(item.id), 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": body
                }
        return json.dumps(line)
    
    # Create JSONL file
    def make_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
                for item in self.items[self.start : self.end]:
                    f.write(self.make_jsonl(item))
                    f.write("\n")

    # Batch mode 
    # Upload JSONL file
    def send_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:
            response = openai.files.create(
                       file=f,
                       purpose="batch"
                                        )
        self.file_id = response.id
    
    # Create batch
    def submit_batch(self):
        response = openai.batches.create(
                   input_file_id=self.file_id,
                   endpoint="/v1/chat/completions",
                   completion_window="24h"
                )
        self.batch_id = response.id

    # Retrieve batch
    def is_ready(self, max_tokens = False):
        pbar = tqdm(desc="Processing batch", bar_format="{desc}: {elapsed}")
        # Backup old batch with 1000 items
        if max_tokens:
            # Use this in case of tokens limits accessed
            self.batch_id = "batch_6a3bd6488b1c8190a72a4f937c7e71b5"

        while True:
            batch = openai.batches.retrieve(self.batch_id)
            pbar.set_description(f"Status: {batch.status}")
            if batch.status == "completed":
                self.output_file_id = batch.output_file_id
                pbar.close()
                return 'Done'
            if batch.status in ("failed", "cancelled", "expired"):
                pbar.close()
                raise RuntimeError(batch.status)
            time.sleep(5)

    # Download items
    def fetch_output(self):
        output_file = str(self.output / self.filename)
        response = openai.files.content(self.output_file_id)
        with open(output_file, "wb") as f:
            f.write(response.read())

    # Preprocessing items 
    # Summary field: content of LLM response
    def apply_output(self):
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                # Make the id as integer
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].summary = summary
        self.done = True

    # For number of batches
    # Create batches
    @classmethod
    def create(cls, items):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    # Run batches
    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls,max_tokens = False):
        for batch in tqdm(cls.batches):
            # Check status
            if not batch.done:
                if batch.is_ready(max_tokens):
                    batch.fetch_output()
                    batch.apply_output()
        # Finished batches            
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    # Save in pickle file
    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    # Load from pickle file
    @classmethod
    def load(cls, items):
        # cls is the class itself
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")

    # Push batches to Hugging Face 
    @classmethod
    def push_to_hub(cls):
        truststore.inject_into_ssl()
        # Push dataset to Hugging Face
        username = "omarshaarawy11"
        # Folder of data
        items_data = f"{username}/best_deal_Batches"
        # Access items inside batches
        items = cls.batches[0].items
        items = [item for item in items if item.summary is not None]
        # Remove the fields that we don't need in the Hugging Face
        for item in items:
            item.full = None
            item.id = None
        percentage = int(len(items) / 100)
        train = items[:percentage * 80]
        val = items[percentage * 80: percentage * 90]
        test = items[percentage * 90: ]
        # Folder then data
        Item.push_to_hub(items_data, train, val, test)