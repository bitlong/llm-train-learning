import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

config = BitsAndBytesConfig(
    # load_in_4bit=True,
    load_in_8bit = True,
)

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    config=config
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")


from peft import PeftModel

# output_dir = "./peft_models"
output_dir = "sql-code-llama/checkpoint-40"
model = PeftModel.from_pretrained(model, output_dir)

print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i)
    _ = torch.tensor([0], device=i)

eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis or nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=200)[0], skip_special_tokens=True))


