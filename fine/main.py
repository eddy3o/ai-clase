import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# -----------------------------
# 1. Cargar dataset
# -----------------------------
dataset = load_dataset("json", data_files="fine/respuestas_fixed.jsonl")

# -----------------------------
# 2. Cargar modelo base
# -----------------------------
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Verificar si se está usando GPU
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU en uso: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Dispositivo del modelo: {model.device}")

# -----------------------------
# 3. Configurar LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 4. Preprocesar el dataset
# -----------------------------
def format_instruction(example):
    prompt = f"Instrucción: {example['instruction']}\nRespuesta:"
    return tokenizer(prompt + example["response"], truncation=True)

tokenized = dataset.map(format_instruction)

# -----------------------------
# 5. Entrenamiento
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora-tutor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    logging_steps=50,
    num_train_epochs=6,
    fp16=True,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# -----------------------------
# 6. Guardar adaptadores LoRA
# -----------------------------
model.save_pretrained("./fine/lora-tutor")
print("Entrenamiento completado. Adaptadores guardados.")