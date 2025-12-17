import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# Low Rank Adaptation (LoRA) para ajuste fino eficiente de modelos de lenguaje grandes

# -----------------------------
# 1. Cargar dataset
# -----------------------------
dataset = load_dataset("json", data_files="Fine-Tunning//tutor_programacion.jsonl")

# -----------------------------
# 2. Cargar modelo base
# -----------------------------
# Usando TinyLlama - modelo abierto, pequeño y gratuito
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Cuantización 8-bit para ahorrar memoria
    device_map="auto",  # Asigna automáticamente a GPU
    torch_dtype="auto"
)

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
    output_dir="./lora-tutor-programacion",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,  # Más frecuente para ver progreso
    num_train_epochs=3,
    fp16=True,  # Precisión mixta para GPU
    save_steps=100,
    max_steps=100,  # Limitar pasos para prueba rápida
    report_to="none",  # Deshabilitar wandb
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
model.save_pretrained("./lora-tutor-programacion")
#tokenizer.save_pretrained("./lora-tutor-programacion")
print("✅ Entrenamiento completado. Adaptadores guardados en: ./lora-tutor-programacion")