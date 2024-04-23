from datetime import datetime
import os
import sys
import transformers

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq


def load_dataset():
    from datasets import load_dataset
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    print(type(dataset))
    # dataset = dataset[:1000]
    # dataset = load_dataset("json", "/app/sql_create_context_v4.json", split="train")
    # train_dataset = dataset.train_test_split(test_size=0.1)["train"]
    # eval_dataset = dataset.train_test_split(test_size=0.1)["test"]
    datasets_split = dataset.train_test_split(train_size=1000, test_size=100)
    eval_dataset = datasets_split["test"]
    train_dataset = datasets_split["train"]

    print(type(train_dataset))
    print(type(eval_dataset))
    return train_dataset, eval_dataset


def load_model():
    base_model = "codellama/CodeLlama-7b-hf"
    from torch.nn import DataParallel

    device_map = "cuda:7"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        # device_map=device_map
        device_map="auto"
    )
    # model = DataParallel(model, device_ids=[6,7]).cuda()
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", device_map=device_map)
    return model, tokenizer


def prompt_input():
    eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis or nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""

    # {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    return model_input


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)


def train_model():
    pass


if __name__ == '__main__':
    train_dataset, eval_dataset = load_dataset()

    print(train_dataset[3])

    model, tokenizer = load_model()
    model_input = prompt_input()
    len(model_input)
    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=False))

    # Tokenization
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    # PEFT 模型微调
    model.train()  # put model back into training mode
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    resume_from_checkpoint = ""  # set this to the adapter_model.bin file you want to resume from

    if resume_from_checkpoint:
        if os.path.exists(resume_from_checkpoint):
            print(f"Restarting from {resume_from_checkpoint}")
            adapters_weights = torch.load(resume_from_checkpoint)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {resume_from_checkpoint} not found")

    wandb_project = ""
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    batch_size = 128  # 128
    per_device_train_batch_size = 32  # 32
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    output_dir = "sql-code-llama"

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,  # 100
        max_steps=400,  # 400
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",  # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True,  # group sequences of roughly the same length together to speed up training
        report_to="none",  # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # if use_wandb else None,

    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    #     model, type(model)
    # )

    if torch.__version__ >= "2" and sys.platform != "win32":
        print("compiling the model")
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained('./peft_models/')

