import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🔹 Cargando modelo base con cuantización 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

print("🔹 Cargando adaptadores LoRA...")
model = PeftModel.from_pretrained(model, "./fine/lora-tutor")

print("🔹 Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
tokenizer.pad_token = tokenizer.eos_token

print("\n✅ Modelo cargado. Listo para hacer inferencia.\n")

# Ejemplo de uso
while True:
    pregunta = input("Tu pregunta (o 'salir' para terminar): ")
    if pregunta.lower() == 'salir':
        break
    
    prompt = f"Instrucción: {pregunta}\nRespuesta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\n🤖 Generando respuesta...\n")
    outputs = model.generate(
        **inputs,
        max_new_tokens=515,
        temperature=0.003,
        do_sample=True,
        top_p=0.85,
        repetition_penalty=1.15,
    )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(respuesta.split("Respuesta:")[-1].strip())
    print("\n" + "-"*50 + "\n")
