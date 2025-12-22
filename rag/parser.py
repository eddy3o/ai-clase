import pandas as pd
import json
from pathlib import Path

def optimizar_para_embeddings(csv_path, output_json=None, incluir_metadata=True):
    """
    Convierte CSV a formato óptimo para embeddings
    
    Args:
        csv_path: ruta al CSV
        output_json: ruta de salida (opcional)
        incluir_metadata: si incluir tema/sentimiento en el texto
    """
    df = pd.read_csv(csv_path)
    
    documentos = []
    
    for idx, row in df.iterrows():
        # Texto base
        texto = row['texto'].strip()
        
        # Opción 1: Solo texto limpio (mejor para embeddings puros)
        doc_simple = {
            'id': row['id'],
            'texto': texto,
            'metadata': {
                'fecha': row['fecha'],
                'tema': row['tema'],
                'sentimiento': row['sentimiento'],
                'likes': row['likes'],
                'repostos': row['reposts']
            }
        }
        
        # Opción 2: Texto enriquecido (mejor para búsqueda semántica)
        texto_enriquecido = f"[{row['tema']}] [{row['sentimiento']}] {texto}"
        
        doc_enriquecido = {
            'id': row['id'],
            'texto': texto_enriquecido,
            'texto_original': texto,
            'metadata': {
                'fecha': row['fecha'],
                'tema': row['tema'],
                'sentimiento': row['sentimiento'],
                'likes': row['likes'],
                'repostos': row['reposts']
            }
        }
        
        # Elegir formato según parámetro
        documentos.append(doc_enriquecido if incluir_metadata else doc_simple)
    
    # Guardar
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(documentos, f, ensure_ascii=False, indent=2)
        print(f"✓ Guardado: {output_json}")
        print(f"✓ Total documentos: {len(documentos)}")
    
    return documentos


def extraer_solo_textos(csv_path, output_txt=None):
    """
    Extrae solo los textos limpios (1 por línea)
    Útil para algunos modelos de embeddings
    """
    df = pd.read_csv(csv_path)
    textos = df['texto'].tolist()
    
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            for texto in textos:
                f.write(texto.strip() + '\n')
        print(f"✓ Guardado: {output_txt}")
        print(f"✓ Total textos: {len(textos)}")
    
    return textos


def crear_corpus_busqueda(csv_path, output_json=None):
    """
    Crea corpus optimizado para búsqueda semántica RAG
    Combina texto con contexto relevante
    """
    df = pd.read_csv(csv_path)
    
    corpus = []
    
    for idx, row in df.iterrows():
        # Formato optimizado para RAG
        documento = {
            'id': f"doc_{row['id']}",
            'content': row['texto'].strip(),
            'enriched_content': f"Tema: {row['tema']}\nSentimiento: {row['sentimiento']}\n\n{row['texto']}",
            'metadata': {
                'fecha': row['fecha'],
                'tema': row['tema'],
                'sentimiento': row['sentimiento'],
                'engagement': row['likes'] + row['reposts'],
                'likes': row['likes'],
                'repostos': row['reposts']
            }
        }
        corpus.append(documento)
    
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print(f"✓ Corpus RAG guardado: {output_json}")
        print(f"✓ Total documentos: {len(corpus)}")
    
    return corpus


def stats_dataset(csv_path):
    """Muestra estadísticas del dataset"""
    df = pd.read_csv(csv_path)
    
    print("=" * 50)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 50)
    print(f"Total documentos: {len(df)}")
    print(f"\nTemas únicos: {df['tema'].nunique()}")
    print(df['tema'].value_counts())
    print(f"\nSentimientos:")
    print(df['sentimiento'].value_counts())
    print(f"\nPromedio de engagement:")
    print(f"  Likes: {df['likes'].mean():.0f}")
    print(f"  Reposts: {df['reposts'].mean():.0f}")
    print(f"\nLongitud promedio texto: {df['texto'].str.len().mean():.0f} caracteres")
    print("=" * 50)


if __name__ == '__main__':
    # ========== CONFIGURACIÓN ==========
    csv_file = 'rack/dataset_sintetico_5000_ampliado.csv'
    output_dir = 'rack/'  
    
    # Mostrar estadísticas
    stats_dataset(csv_file)
    
    print("\n🚀 GENERANDO FORMATOS...\n")
    
    # Formato 1: Textos simples
    optimizar_para_embeddings(
        csv_file, 
        f'{output_dir}corpus_simple.json',
        incluir_metadata=False
    )
    
    # Formato 2: Textos enriquecidos
    optimizar_para_embeddings(
        csv_file,
        f'{output_dir}corpus_enriquecido.json',
        incluir_metadata=True
    )
    
    # Formato 3: Solo textos planos
    extraer_solo_textos(
        csv_file,
        f'{output_dir}textos_planos.txt'
    )
    
    # Formato 4: Corpus para RAG
    crear_corpus_busqueda(
        csv_file,
        f'{output_dir}corpus_rag.json'
    )
    
    print("\n✅ LISTO! Formatos generados:")
    print("  • corpus_simple.json      → Texto + metadata separada")
    print("  • corpus_enriquecido.json → Texto con tema/sentimiento")
    print("  • textos_planos.txt       → Solo textos (1 por línea)")
    print("  • corpus_rag.json         → Formato optimizado RAG")
