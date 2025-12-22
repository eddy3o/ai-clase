import argparse
import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Convertir LoRA a GGUF para Ollama")
    parser.add_argument("--base", required=True, help="Ruta del modelo base (llama3)")
    parser.add_argument("--model", required=True, help="Ruta del LoRA entrenado")
    parser.add_argument("--out", required=True, help="Archivo GGUF de salida")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    print("🔹 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print("🔹 Cargando modelo base en GPU con cuantización 4-bit...")
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "6GB"}  # Limita uso de VRAM
    )

    print("🔹 Cargando adaptadores LoRA...")
    model = PeftModel.from_pretrained(model, args.model)

    print("🔹 Fusionando LoRA con el modelo base...")
    model = model.merge_and_unload()

    tmp_dir = "./fine/merged_model"
    os.makedirs(tmp_dir, exist_ok=True)

    print("🔹 Guardando modelo fusionado (en formato cuantizado)...")
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)

    print("\n✅ Modelo fusionado guardado en:", tmp_dir)
    print("⚠️  NOTA: El modelo está en formato cuantizado.")
    print("    Para usarlo con Transformers: usa PeftModel.from_pretrained()")
    print("    Para GGUF: llama.cpp podría no soportar el formato cuantizado.")
    print("\n🔹 Intentando conversión a GGUF...")
    
    llama_cpp_convert = "./fine/llama.cpp/convert_hf_to_gguf.py"

    try:
        subprocess.run([
            "python",
            llama_cpp_convert,
            tmp_dir,
            "--outfile",
            args.out,
            "--outtype",
            "f16"
        ], check=True)
        print(f"✅ Conversión GGUF completa: {args.out}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ La conversión a GGUF falló (esperado con modelos cuantizados)")
        print(f"   Puedes usar el modelo fusionado directamente desde: {tmp_dir}")
        print(f"   Carga con: AutoModelForCausalLM.from_pretrained('{tmp_dir}')")


if __name__ == "__main__":
    main()