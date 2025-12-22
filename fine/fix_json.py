import json

# Leer todas las líneas
with open('respuestas.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
problem_lines = []

for i, line in enumerate(lines, 1):
    if not line.strip():
        continue
    
    # La línea 201 y otras similares tienen template literals con backticks
    # En JSON no se pueden escapar backticks, simplemente se dejan sin escapar
    # O mejor aún, los reemplazamos con comillas simples para evitar confusión
    
    fixed_line = line
    
    # Si la línea contiene \`, es inválida
    if r'\`' in fixed_line:
        # Opción 1: Reemplazar backticks con comillas simples
        # Los template literals de TypeScript no son necesarios mantenerlos como backticks en texto
        fixed_line = fixed_line.replace('`', "'")
        fixed_line = fixed_line.replace(r"\'", "'")  # Por si acaso
        problem_lines.append(i)
    
    try:
        # Verificar que sea JSON válido
        data = json.loads(fixed_line)
        # Re-serializar para asegurar formato correcto
        fixed_lines.append(json.dumps(data, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"ERROR en línea {i}: {e}")
        print(f"Contenido: {line[:100]}...")
        continue

print(f"\nLíneas con problemas corregidas: {problem_lines}")
print(f"Total de líneas procesadas: {len(fixed_lines)}")

# Guardar archivo corregido
with open('respuestas_fixed.jsonl', 'w', encoding='utf-8') as f:
    for line in fixed_lines:
        f.write(line + '\n')

print("Archivo guardado como 'respuestas_fixed.jsonl'")

# Verificar el archivo generado
print("\nVerificando archivo corregido...")
with open('respuestas_fixed.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"ERROR en línea {i} del archivo corregido: {e}")

print("Verificación completada.")
