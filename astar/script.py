import pygame
import math
from queue import PriorityQueue

# ==================== CONFIGURACIÓN ====================

pygame.init()

info_pantalla = pygame.display.Info()
ANCHO_BASE = min(int(info_pantalla.current_w * 0.9), 1600)
ALTO_BASE = min(int(info_pantalla.current_h * 0.9), 1000)
PANEL_INFO = 250
HEADER_ALTURA = 0

VENTANA = pygame.display.set_mode((ANCHO_BASE, ALTO_BASE), pygame.RESIZABLE)
pygame.display.set_caption("A* PATHFINDER")

# Fuentes
FUENTE_TITULO = pygame.font.Font(None, 32)
FUENTE_NORMAL = pygame.font.Font(None, 20)
FUENTE_PEQUEÑA = pygame.font.Font(None, 16)
FUENTE_NODO = pygame.font.Font(None, 14)

COLORES = {
    'fondo': (8, 8, 20),
    'grid': (0, 40, 60),
    'pared': (20, 20, 35),
    'pared_borde': (100, 50, 150),
    'inicio': (0, 255, 255),
    'fin': (255, 0, 255),
    'camino': (0, 255, 150),
    'visitado': (80, 0, 100),
    'visitado_borde': (150, 0, 200),
    'frontera': (255, 150, 0),
    'procesando': (255, 255, 0),
    'panel': (10, 10, 25),
    'panel_borde': (0, 150, 255),
    'texto': (0, 255, 255),
    'texto_claro': (0, 180, 220),
    'acento': (255, 0, 255),
    'acento2': (255, 200, 0)
}

# ==================== CLASE NODO ====================

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = 0
        self.y = 0
        self.ancho = ancho
        self.total_filas = total_filas
        self.color = COLORES['fondo']
        self.vecinos = []
        
        # g: costo real desde inicio, h: heurística al objetivo, f: g + h
        self.costo_g = float('inf')
        self.costo_h = 0
        self.costo_f = float('inf')
        self.nodo_padre = None

    def obtener_posicion(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == COLORES['pared']

    def es_inicio(self):
        return self.color == COLORES['inicio']

    def es_fin(self):
        return self.color == COLORES['fin']

    def resetear_nodo(self):
        self.color = COLORES['fondo']
        self.costo_g = float('inf')
        self.costo_h = 0
        self.costo_f = float('inf')
        self.nodo_padre = None

    def establecer_inicio(self):
        self.color = COLORES['inicio']

    def establecer_pared(self):
        self.color = COLORES['pared']

    def establecer_fin(self):
        self.color = COLORES['fin']

    def establecer_camino(self):
        self.color = COLORES['camino']

    def establecer_visitado(self):
        self.color = COLORES['visitado']

    def establecer_frontera(self):
        self.color = COLORES['frontera']
    
    def establecer_procesando(self):
        self.color = COLORES['procesando']

    def renderizar(self, ventana, mostrar_costos=False):
        if self.es_pared():
            pygame.draw.rect(ventana, self.color, (self.x + 2, self.y + 2, self.ancho - 4, self.ancho - 4))
            pygame.draw.rect(ventana, COLORES['pared_borde'], (self.x + 1, self.y + 1, self.ancho - 2, self.ancho - 2), 2)
        
        elif self.es_inicio():
            pygame.draw.rect(ventana, self.color, (self.x + 3, self.y + 3, self.ancho - 6, self.ancho - 6), border_radius=4)
        
        elif self.es_fin():
            pygame.draw.rect(ventana, self.color, (self.x + 3, self.y + 3, self.ancho - 6, self.ancho - 6), border_radius=4)
        
        elif self.color == COLORES['camino']:
            pygame.draw.rect(ventana, self.color, (self.x + 3, self.y + 3, self.ancho - 6, self.ancho - 6), border_radius=3)
        
        elif self.color == COLORES['frontera']:
            pygame.draw.rect(ventana, self.color, (self.x + 2, self.y + 2, self.ancho - 4, self.ancho - 4))
        
        elif self.color == COLORES['visitado']:
            pygame.draw.rect(ventana, self.color, (self.x + 2, self.y + 2, self.ancho - 4, self.ancho - 4))
        
        elif self.color == COLORES['procesando']:
            pygame.draw.rect(ventana, self.color, (self.x + 3, self.y + 3, self.ancho - 6, self.ancho - 6), border_radius=4)
            pygame.draw.rect(ventana, (255, 255, 255), (self.x + 1, self.y + 1, self.ancho - 2, self.ancho - 2), 2, border_radius=4)
        
        # Solo mostrar costos si el nodo es suficientemente grande
        if mostrar_costos and self.ancho > 40 and self.color in [COLORES['frontera'], COLORES['visitado'], COLORES['camino']]:
            if self.costo_g != float('inf'):
                g_texto = FUENTE_NODO.render(f"g:{self.costo_g:.1f}", True, COLORES['texto_claro'])
                f_texto = FUENTE_NODO.render(f"f:{self.costo_f:.1f}", True, COLORES['acento2'])
                
                ventana.blit(g_texto, (self.x + 5, self.y + 8))
                ventana.blit(f_texto, (self.x + 5, self.y + self.ancho - 20))

    def calcular_vecinos(self, grid):
        self.vecinos = []
        # (fila_delta, col_delta, costo)
        movimientos = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
        ]
        
        for df, dc, costo in movimientos:
            nf, nc = self.fila + df, self.col + dc
            
            if 0 <= nf < self.total_filas and 0 <= nc < self.total_filas:
                vecino = grid[nf][nc]
                if not vecino.es_pared():
                    # Bloquear diagonales si ambos lados están obstruidos
                    if abs(df) == 1 and abs(dc) == 1:
                        if not grid[self.fila + df][self.col].es_pared() or \
                           not grid[self.fila][self.col + dc].es_pared():
                            self.vecinos.append((vecino, costo))
                    else:
                        self.vecinos.append((vecino, costo))

    def __lt__(self, other):
        return self.costo_f < other.costo_f

# ==================== HEURÍSTICA ====================

def obtener_distancia_euclidiana(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# ==================== ALGORITMO A* ====================

def buscar_camino_a_estrella(grid, inicio, fin, funcion_dibujar, epsilon=1.2, 
                        mostrar_costos=False, velocidad=1):
    # epsilon controla balance optimalidad/velocidad (1.0=óptimo, >1=más voraz)
    # velocidad: 0=paso a paso con ESPACIO, >0=auto con pausa entre N nodos
    
    contador = 0
    cola_abierta = PriorityQueue()
    
    inicio.costo_g = 0
    inicio.costo_h = obtener_distancia_euclidiana(inicio.obtener_posicion(), fin.obtener_posicion())
    inicio.costo_f = inicio.costo_g + epsilon * inicio.costo_h
    
    cola_abierta.put((inicio.costo_f, contador, inicio))
    conjunto_abierto = {inicio}
    conjunto_cerrado = set()
    
    nodos_explorados = 0
    info_actual = []
    pausado = False
    nodo_procesando_actual = None
    paso_completado = velocidad > 0

    while not cola_abierta.empty():
        # Esperar input del usuario en modo manual o pausado
        while (velocidad == 0 and not paso_completado) or pausado:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_n:
                        paso_completado = True
                    elif event.key == pygame.K_p:
                        pausado = not pausado
                        if not pausado:
                            break
            
            if velocidad == 0 or pausado:
                funcion_dibujar(info_actual, mostrar_costos, nodos_explorados)
                pygame.time.delay(50)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pausado = not pausado
        
        if nodo_procesando_actual and nodo_procesando_actual != inicio:
            nodo_procesando_actual.establecer_visitado()
        
        nodo_actual = cola_abierta.get()[2]
        
        if nodo_actual in conjunto_cerrado:
            continue
        
        if nodo_actual in conjunto_abierto:
            conjunto_abierto.remove(nodo_actual)
        
        conjunto_cerrado.add(nodo_actual)
        nodos_explorados += 1
        
        if nodo_actual != inicio and nodo_actual != fin:
            nodo_actual.establecer_procesando()
        nodo_procesando_actual = nodo_actual
        
        paso_completado = False

        if nodo_actual == fin:
            camino = construir_ruta(inicio, fin, funcion_dibujar)
            fin.establecer_fin()
            inicio.establecer_inicio()
            
            info_final = [
                "═══ CAMINO ENCONTRADO ═══",
                f"Longitud: {len(camino)} nodos",
                f"Costo: {fin.costo_g:.2f}",
                f"Explorados: {nodos_explorados}",
                ""
            ]
            
            for i, nodo in enumerate(camino[:20]):  # Solo primeros 20
                info_final.append(f"{i+1}. ({nodo.fila},{nodo.col})")
            
            if len(camino) > 20:
                info_final.append(f"... ({len(camino)-20} más)")
            
            return info_final

        # Revisar cada vecino y actualizar costos si encontramos mejor camino
        for vecino, costo_movimiento in nodo_actual.vecinos:
            if vecino in conjunto_cerrado:
                continue
            
            temp_g = nodo_actual.costo_g + costo_movimiento
            
            if temp_g < vecino.costo_g:
                vecino.nodo_padre = nodo_actual
                vecino.costo_g = temp_g
                vecino.costo_h = obtener_distancia_euclidiana(vecino.obtener_posicion(), fin.obtener_posicion())
                vecino.costo_f = temp_g + epsilon * vecino.costo_h
                
                contador += 1
                cola_abierta.put((vecino.costo_f, contador, vecino))
                
                if vecino not in conjunto_abierto:
                    conjunto_abierto.add(vecino)
                    vecino.establecer_frontera()
                
                if velocidad == 0 or (velocidad > 0 and nodos_explorados % velocidad == 0):
                    modo = "MANUAL" if velocidad == 0 else ("PAUSADO" if pausado else "AUTO")
                    info_actual = [
                        f"Modo: {modo}",
                        f"Procesando: ({nodo_actual.fila},{nodo_actual.col})",
                        f"Explorados: {nodos_explorados}",
                        f"Frontera: {len(conjunto_abierto)}"
                    ]
                    if velocidad == 0:
                        info_actual.insert(1, ">> ESPACIO: siguiente paso")
                    elif pausado:
                        info_actual.insert(1, ">> P: continuar")
                    funcion_dibujar(info_actual, mostrar_costos, nodos_explorados)

        if velocidad == 0 or (velocidad > 0 and nodos_explorados % velocidad == 0):
            modo = "MANUAL" if velocidad == 0 else ("PAUSADO" if pausado else "AUTO")
            if not info_actual:
                info_actual = [
                    f"Modo: {modo}",
                    f"Procesando: ({nodo_actual.fila},{nodo_actual.col})",
                    f"Explorados: {nodos_explorados}"
                ]
            funcion_dibujar(info_actual, mostrar_costos, nodos_explorados)

    return ["✗ No se encontró camino", f"Explorados: {nodos_explorados}"]

def construir_ruta(inicio, fin, funcion_dibujar):
    camino = []
    nodo_actual = fin
    
    while nodo_actual != inicio:
        if nodo_actual.nodo_padre is None:
            break
        nodo_actual = nodo_actual.nodo_padre
        if nodo_actual != inicio:
            nodo_actual.establecer_camino()
            camino.insert(0, nodo_actual)
        pygame.time.delay(30)
        funcion_dibujar([], False)
    
    return [inicio] + camino + [fin]

# ==================== INTERFAZ ====================

def inicializar_grilla(filas, ancho_disponible):
    grid = []
    ancho_nodo = ancho_disponible // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def renderizar_lineas_grilla(ventana, filas, offset_x, offset_y, ancho_nodo):
    for i in range(filas + 1):
        pygame.draw.line(ventana, COLORES['grid'],
                        (offset_x, offset_y + i * ancho_nodo),
                        (offset_x + filas * ancho_nodo, offset_y + i * ancho_nodo))
        pygame.draw.line(ventana, COLORES['grid'],
                        (offset_x + i * ancho_nodo, offset_y),
                        (offset_x + i * ancho_nodo, offset_y + filas * ancho_nodo))

def mostrar_header_superior(ventana, ancho_ventana, epsilon, velocidad, nodos_explorados=0):
    pass

def mostrar_panel_lateral(ventana, ancho_ventana, alto_ventana, info_estado, scroll, epsilon, velocidad):
    x_panel = ancho_ventana - PANEL_INFO
    y_inicio = HEADER_ALTURA
    
    pygame.draw.rect(ventana, COLORES['panel'], (x_panel, y_inicio, PANEL_INFO, alto_ventana - y_inicio))
    pygame.draw.line(ventana, COLORES['panel_borde'], (x_panel, y_inicio), (x_panel, alto_ventana), 2)
    
    y = y_inicio + 10
    pygame.draw.rect(ventana, (15, 15, 30), (x_panel + 8, y, PANEL_INFO - 16, 160))
    pygame.draw.rect(ventana, COLORES['panel_borde'], (x_panel + 8, y, PANEL_INFO - 16, 160), 1)
    
    titulo = FUENTE_NORMAL.render("CONTROLES", True, COLORES['texto'])
    ventana.blit(titulo, (x_panel + 15, y + 6))
    
    y += 28
    controles = [
        ("ENTER", "Iniciar", COLORES['inicio']),
        ("ESPACIO", "Paso", COLORES['acento2']),
        ("P", "Pausa", COLORES['acento']),
        ("R", "Reset", COLORES['texto_claro']),
        ("M", "Costos", COLORES['texto_claro']),
        ("W/S", "Velocidad", COLORES['frontera']),
        ("A/D", "Epsilon", COLORES['camino']),
    ]
    
    for tecla, desc, color in controles:
        texto = FUENTE_PEQUEÑA.render(f"{tecla}: {desc}", True, color)
        ventana.blit(texto, (x_panel + 15, y))
        y += 18
    
    y += 12
    
    if info_estado:
        modulo_h = alto_ventana - y - 10
        pygame.draw.rect(ventana, (0, 5, 10), (x_panel + 8, y, PANEL_INFO - 16, modulo_h))
        pygame.draw.rect(ventana, COLORES['inicio'], (x_panel + 8, y, PANEL_INFO - 16, modulo_h), 1)
        
        titulo = FUENTE_NORMAL.render("CONSOLA", True, COLORES['texto'])
        ventana.blit(titulo, (x_panel + 15, y + 6))
        
        y += 28
        y_inicio_scroll = y
        area_scroll = alto_ventana - y_inicio_scroll - 12
        
        for i, linea in enumerate(info_estado):
            y_linea = y_inicio_scroll + (i * 16) - scroll
            if y_inicio_scroll <= y_linea < y_inicio_scroll + area_scroll:
                if "═══" in str(linea) or "ENCONTRADO" in str(linea):
                    color = COLORES['acento']
                elif "Procesando" in str(linea):
                    color = COLORES['frontera']
                else:
                    color = COLORES['texto_claro']
                
                texto = pygame.font.Font(None, 14).render(str(linea), True, color)
                ventana.blit(texto, (x_panel + 15, y_linea))
        
        # Scrollbar proporcional al contenido
        contenido_h = len(info_estado) * 16
        if contenido_h > area_scroll:
            scroll_max = contenido_h - area_scroll
            thumb_h = max(20, int((area_scroll / contenido_h) * area_scroll))
            thumb_y = y_inicio_scroll + int((scroll / scroll_max) * (area_scroll - thumb_h))
            
            pygame.draw.rect(ventana, COLORES['acento'], (x_panel + PANEL_INFO - 14, thumb_y, 5, thumb_h))
            
            return scroll_max
    
    return 0

def actualizar_ventana(ventana, grid, filas, ancho_ventana, alto_ventana, info_estado, scroll, 
           epsilon, velocidad, mostrar_costos, nodos_explorados=0):
    ventana.fill(COLORES['fondo'])
    
    mostrar_header_superior(ventana, ancho_ventana, epsilon, velocidad, nodos_explorados)
    
    ancho_juego = ancho_ventana - PANEL_INFO
    alto_juego = alto_ventana - HEADER_ALTURA
    ancho_nodo = min(ancho_juego, alto_juego) // filas
    
    offset_x = (ancho_juego - (ancho_nodo * filas)) // 2
    offset_y = HEADER_ALTURA + (alto_juego - (ancho_nodo * filas)) // 2
    
    for fila in grid:
        for nodo in fila:
            nodo.x = offset_x + nodo.col * ancho_nodo
            nodo.y = offset_y + nodo.fila * ancho_nodo
            nodo.ancho = ancho_nodo
            nodo.renderizar(ventana, mostrar_costos)
    
    renderizar_lineas_grilla(ventana, filas, offset_x, offset_y, ancho_nodo)
    
    scroll_max = mostrar_panel_lateral(ventana, ancho_ventana, alto_ventana, info_estado, 
                                     scroll, epsilon, velocidad)
    
    pygame.display.update()
    return offset_x, offset_y, ancho_nodo, scroll_max

def convertir_click_a_posicion(pos, filas, offset_x, offset_y, ancho_nodo):
    x, y = pos
    x -= offset_x
    y -= offset_y
    
    if x < 0 or y < 0:
        return None, None
    
    col = x // ancho_nodo
    fila = y // ancho_nodo
    
    if fila >= filas or col >= filas:
        return None, None
    
    return fila, col

# ==================== MAIN ====================

def main():
    FILAS = 11
    grid = inicializar_grilla(FILAS, ANCHO_BASE - PANEL_INFO)
    
    inicio = None
    fin = None
    info_estado = None
    scroll = 0
    
    epsilon = 1.2
    velocidad = 5
    mostrar_costos = True
    
    ancho_ventana = ANCHO_BASE
    alto_ventana = ALTO_BASE
    
    corriendo = True
    clock = pygame.time.Clock()

    while corriendo:
        clock.tick(60)
        
        offset_x, offset_y, ancho_nodo, scroll_max = actualizar_ventana(
            VENTANA, grid, FILAS, ancho_ventana, alto_ventana, 
            info_estado, scroll, epsilon, velocidad, mostrar_costos
        )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False
            
            if event.type == pygame.VIDEORESIZE:
                ancho_ventana = event.w
                alto_ventana = event.h
            
            if event.type == pygame.MOUSEWHEEL:
                scroll = max(0, min(scroll - event.y * 20, scroll_max))
            
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if pos[0] < ancho_ventana - PANEL_INFO:
                    fila, col = convertir_click_a_posicion(pos, FILAS, offset_x, offset_y, ancho_nodo)
                    if fila is not None:
                        nodo = grid[fila][col]
                        if not inicio and nodo != fin:
                            inicio = nodo
                            inicio.establecer_inicio()
                        elif not fin and nodo != inicio:
                            fin = nodo
                            fin.establecer_fin()
                        elif nodo != inicio and nodo != fin:
                            nodo.establecer_pared()
            
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                if pos[0] < ancho_ventana - PANEL_INFO:
                    fila, col = convertir_click_a_posicion(pos, FILAS, offset_x, offset_y, ancho_nodo)
                    if fila is not None:
                        nodo = grid[fila][col]
                        nodo.resetear_nodo()
                        if nodo == inicio:
                            inicio = None
                        elif nodo == fin:
                            fin = None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            if not nodo.es_inicio() and not nodo.es_fin() and not nodo.es_pared():
                                nodo.resetear_nodo()
                            nodo.calcular_vecinos(grid)
                    
                    def wrapper_dibujar(info, costos, nodos_exp=0):
                        nonlocal info_estado
                        info_estado = info
                        actualizar_ventana(VENTANA, grid, FILAS, ancho_ventana, alto_ventana, 
                               info_estado, scroll, epsilon, velocidad, costos, nodos_exp)
                    
                    resultado = buscar_camino_a_estrella(grid, inicio, fin, wrapper_dibujar, 
                                                    epsilon, mostrar_costos, velocidad)
                    if resultado:
                        info_estado = resultado
                
                elif event.key == pygame.K_r:
                    grid = inicializar_grilla(FILAS, ancho_ventana - PANEL_INFO)
                    inicio = None
                    fin = None
                    info_estado = None
                    scroll = 0
                
                elif event.key == pygame.K_w:
                    velocidad = min(velocidad + 1, 20)
                elif event.key == pygame.K_s:
                    velocidad = max(velocidad - 1, 0)
                
                elif event.key == pygame.K_d:
                    epsilon = min(epsilon + 0.2, 5.0)
                elif event.key == pygame.K_a:
                    epsilon = max(epsilon - 0.2, 1.0)
                
                elif event.key == pygame.K_m:
                    mostrar_costos = not mostrar_costos

    pygame.quit()

if __name__ == '__main__':
    main()
