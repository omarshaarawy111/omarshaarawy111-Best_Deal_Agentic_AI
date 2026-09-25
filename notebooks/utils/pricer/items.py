from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self

# Configurations
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

# Validation class for each product
# title, category, full, weight coming from parser
# id coming from preprocessing phase
# Summary field comes from LLM respond in curating phase
class Item(BaseModel):
    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    # Return value of Item class to be an Item object 
    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"


    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):\
        # Return it as DatasetDict object then push to hub
        train_ds = Dataset.from_list([item.model_dump() for item in train])
        val_ds   = Dataset.from_list([item.model_dump() for item in val])
        test_ds  = Dataset.from_list([item.model_dump() for item in test])
        DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }).push_to_hub(dataset_name)
        
    # Load dataset to Hugging Face
    # return dataset ---> validate
    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        # Return it as DatasetDict object for the three lists ---> (train, validation and test) as we push it with three lists before
        # Any loading other datasets from other owners should contain these three lists
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
    
    # Count tokens in the summary field
    def count_tokens(self, tokenizer):
        return len(tokenizer.encode(self.summary, add_special_tokens=False))
    
    # Create prompt and completion features
    def make_prompts(self, tokenizer, max_tokens, do_round):
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = self.summary
        # Input    
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        # Output
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    # Count the total number of tokens in the prompt and completion fields
    def count_prompt_tokens(self, tokenizer):
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)

    # Convert the prompt and completion fields to a dictionary format for pushing to Hugging Face Hub 
    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    # Push prompt and completion dataset to Hugging Face
    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ):
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)


    # Load dataset to Hugging Face
    @classmethod
    def prompt_from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        # Return it as DatasetDict object for the three lists ---> (train, validation and test) as we push it with three lists before
        # Any loading other datasets from other owners should contain these three lists
        return (
            [row for row in ds["train"]],
            [row for row in ds["val"]],
            [row for row in ds["test"]],
        )    