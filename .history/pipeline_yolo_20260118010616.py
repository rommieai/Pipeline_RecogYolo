"""
Pipeline YOLOv11 con anotación manual interactiva para detectar MÚLTIPLES CLASES
CON SOPORTE PARA MÚLTIPLES BOUNDING BOXES POR IMAGEN
Estructura esperada:
  input_files/
    ├── contador/
    │   ├── imagen1.jpg
    │   └── ...
    ├── logo/
    │   ├── imagen1.jpg
    │   └── ...
    └── caja/
        ├── imagen1.jpg
        └── ...
"""

import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import yaml
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
import json
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    print("⚠️  pillow-heif no instalado. Para soporte HEIC: pip install pillow-heif")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

INPUT_FOLDERS = {
    0: "input_files/logo",
    1: "input_files/contador",
    2: "input_files/caja"
}

CLASSES = {
    0: 'logo',
    1: 'contador',
    2: 'caja'
}

PROJECT_NAME = "mercadolibre_detection_final"
ANNOTATIONS_FILE = "annotations_multiclass.json"
TRAIN_SPLIT = 0.8

# ============================================================================
# ANOTADOR INTERACTIVO MULTI-CLASE CON MÚLTIPLES BOUNDING BOXES
# ============================================================================

class MultiClassImageAnnotator:
    """Interfaz gráfica para anotar múltiples clases y múltiples objetos por imagen"""
    
    def __init__(self, images_by_class, all_annotations):
        # Filtrar imágenes sin anotar por clase
        self.images_by_class = {}
        self.unannotated_count = 0
        
        for class_id, image_paths in images_by_class.items():
            unannotated = [
                img for img in image_paths 
                if str(img) not in all_annotations
            ]
            self.images_by_class[class_id] = unannotated
            self.unannotated_count += len(unannotated)
        
        self.all_annotations = all_annotations
        self.current_class = None
        self.current_index = 0
        
        # Verificar si hay imágenes sin anotar
        if self.unannotated_count == 0:
            total_annotated = len(all_annotations)
            print("\n✅ ¡Todas las imágenes ya están anotadas!")
            print(f"   Total: {total_annotated} anotaciones")
            for class_id, class_name in CLASSES.items():
                count = sum(1 for ann in all_annotations.values() if ann['class'] == class_id)
                print(f"   - {class_name}: {count}")
            self.root = None
            return
        
        # Mostrar estadísticas
        print(f"\n📊 Estado de anotaciones:")
        total_images = sum(len(imgs) for imgs in images_by_class.values())
        print(f"   ✅ Ya anotadas: {len(all_annotations)}")
        print(f"   ❌ Sin anotar: {self.unannotated_count}")
        print(f"   📁 Total: {total_images}")
        
        for class_id, class_name in CLASSES.items():
            unannotated = len(self.images_by_class.get(class_id, []))
            total_class = len(images_by_class.get(class_id, []))
            annotated = total_class - unannotated
            print(f"   - {class_name}: {annotated}/{total_class} anotadas")
        
        # Encontrar primera clase con imágenes sin anotar
        for class_id in sorted(CLASSES.keys()):
            if self.images_by_class.get(class_id):
                self.current_class = class_id
                break
        
        if self.current_class is None:
            print("\n✅ Todas las clases están completamente anotadas")
            self.root = None
            return
        
        # Configurar ventana principal
        self.root = tk.Tk()
        self.root.title(f"Anotador Multi-Bbox YOLOv11 (Pendientes: {self.unannotated_count})")
        self.root.geometry("1100x850")
        
        # Variables para el rectángulo
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_bboxes = []  # Lista de bboxes de la imagen actual
        
        # Label de información superior
        self.info_label = tk.Label(self.root, text="", font=('Arial', 11, 'bold'), 
                                   bg='lightblue', pady=5)
        self.info_label.pack(side=tk.TOP, fill=tk.X)
        
        # Frame de botones SUPERIOR
        button_frame_top = tk.Frame(self.root, bg='lightgray', pady=8)
        button_frame_top.pack(side=tk.TOP, fill=tk.X)
        
        # Botones de navegación
        tk.Button(button_frame_top, text="⬅️ Anterior", command=self.prev_image, 
                 font=('Arial', 11, 'bold'), padx=15, pady=5).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame_top, text="Siguiente ➡️", command=self.next_image,
                 font=('Arial', 11, 'bold'), padx=15, pady=5).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame_top, text="🗑️ Borrar Último", command=self.delete_last_bbox,
                 font=('Arial', 11), padx=15, pady=5, bg='#ffcccc').pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame_top, text="🗑️ Borrar Todo", command=self.clear_all_annotations,
                 font=('Arial', 11), padx=15, pady=5, bg='#ff9999').pack(side=tk.LEFT, padx=5)
        
        # Selector de clase PARA EL PRÓXIMO BBOX
        class_frame = tk.Frame(button_frame_top, bg='lightgray')
        class_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(class_frame, text="Próximo bbox:", font=('Arial', 10, 'bold'), 
                bg='lightgray').pack(side=tk.LEFT, padx=5)
        
        self.class_buttons = {}
        self.selected_class_for_bbox = self.current_class  # Clase para el próximo bbox
        for class_id, class_name in CLASSES.items():
            btn = tk.Button(class_frame, text=class_name.upper(), 
                          command=lambda cid=class_id: self.set_bbox_class(cid),
                          font=('Arial', 10), padx=10, pady=3)
            btn.pack(side=tk.LEFT, padx=2)
            self.class_buttons[class_id] = btn
        
        tk.Button(button_frame_top, text="💾 Guardar y Salir", command=self.save_and_exit,
                 font=('Arial', 11, 'bold'), padx=15, pady=5, bg='#90EE90').pack(side=tk.RIGHT, padx=5)
        
        # Canvas para mostrar imagen
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='gray')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Label de instrucciones
        instructions = tk.Label(self.root, 
                               text=f"📝 Instrucciones: Selecciona clase con botones (1/2/3) ANTES de dibujar | "
                                    f"Arrastra para crear bbox | Cada bbox puede ser de clase diferente | "
                                    f"Colores: 🟢=Logo 🟡=Contador 🔵=Caja | "
                                    f"Clic 'Siguiente' cuando termines | {self.unannotated_count} pendientes",
                               font=('Arial', 9), bg='lightyellow', pady=3, wraplength=1050)
        instructions.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Eventos del mouse
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Atajos de teclado
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.delete_last_bbox())
        self.root.bind("<Escape>", lambda e: self.save_and_exit())
        self.root.bind("1", lambda e: self.set_bbox_class(0))
        self.root.bind("2", lambda e: self.set_bbox_class(1))
        self.root.bind("3", lambda e: self.set_bbox_class(2))
        
        # Cargar primera imagen
        self.update_class_buttons()
        self.load_image()
    
    def set_bbox_class(self, class_id):
        """Establece la clase para el próximo bbox a dibujar"""
        self.selected_class_for_bbox = class_id
        self.update_class_buttons()
        
        class_name = CLASSES[class_id]
        print(f"🎯 Próximo bbox será de clase: {class_name.upper()}")
    
    def get_class_color(self, class_id):
        """Retorna el color según la clase"""
        colors = {
            0: '#00FF00',  # Verde - Logo
            1: '#FFD700',  # Amarillo/Dorado - Contador  
            2: '#1E90FF'   # Azul - Caja
        }
        return colors.get(class_id, 'green')
    
    def switch_class(self, class_id):
        """Cambia a otra clase"""
        if class_id not in self.images_by_class or not self.images_by_class[class_id]:
            messagebox.showinfo("Clase completa", 
                              f"La clase '{CLASSES[class_id]}' ya está completamente anotada")
            return
        
        self.current_class = class_id
        self.current_index = 0
        self.update_class_buttons()
        self.load_image()
    
    def update_class_buttons(self):
        """Actualiza el estilo de los botones de clase"""
        for class_id, btn in self.class_buttons.items():
            if class_id == self.selected_class_for_bbox:
                btn.config(bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'))
            else:
                btn.config(bg='SystemButtonFace', fg='black', font=('Arial', 10))
    
    def save_annotations(self):
        """Guarda todas las anotaciones en archivo JSON"""
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(self.all_annotations, f, indent=2)
        
        total_annotated = len(self.all_annotations)
        print(f"💾 Guardadas {total_annotated} anotaciones")
    
    def load_image(self):
        """Carga y muestra la imagen actual"""
        images = self.images_by_class.get(self.current_class, [])
        
        if not images or self.current_index >= len(images):
            # Buscar siguiente clase con imágenes
            next_class = None
            for class_id in sorted(CLASSES.keys()):
                if class_id > self.current_class and self.images_by_class.get(class_id):
                    next_class = class_id
                    break
            
            if next_class is None:
                messagebox.showinfo("Completado", 
                                  "¡Todas las imágenes pendientes han sido anotadas!")
                self.save_and_exit()
                return
            
            self.current_class = next_class
            self.current_index = 0
            self.update_class_buttons()
            self.load_image()
            return
        
        # Obtener ruta actual
        img_path = images[self.current_index]
        self.current_image_path = str(img_path)
        
        # Cargar imagen
        try:
            self.pil_image = Image.open(img_path)
            if self.pil_image.mode not in ('RGB', 'L'):
                self.pil_image = self.pil_image.convert('RGB')
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")
            self.next_image()
            return
        
        self.original_width, self.original_height = self.pil_image.size
        
        # Redimensionar si es muy grande
        max_display_size = 900
        if self.original_width > max_display_size or self.original_height > max_display_size:
            self.pil_image.thumbnail((max_display_size, max_display_size), Image.Resampling.LANCZOS)
        
        self.display_width, self.display_height = self.pil_image.size
        
        # Calcular escala
        self.scale_x = self.original_width / self.display_width
        self.scale_y = self.original_height / self.display_height
        
        # Convertir a ImageTk
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        
        # Actualizar canvas
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Cargar bboxes existentes
        if self.current_image_path in self.all_annotations:
            self.current_bboxes = self.all_annotations[self.current_image_path].copy()
        else:
            self.current_bboxes = []
        
        # Dibujar anotaciones existentes
        self.redraw_all_bboxes()
        
        # Actualizar información
        class_name = CLASSES[self.current_class]
        annotated = "✅" if self.current_image_path in self.all_annotations else "❌"
        total_annotated = len(self.all_annotations)
        
        remaining_in_class = len(images) - self.current_index - (1 if self.current_image_path in self.all_annotations else 0)
        bbox_count = len(self.current_bboxes)
        
        # Contar bboxes por clase en esta imagen
        class_counts = {class_id: 0 for class_id in CLASSES.keys()}
        for bbox in self.current_bboxes:
            class_counts[bbox['class']] += 1
        
        counts_str = " | ".join([f"{CLASSES[cid]}:{cnt}" for cid, cnt in class_counts.items() if cnt > 0])
        if not counts_str:
            counts_str = "ninguno"
        
        next_class_name = CLASSES[self.selected_class_for_bbox]
        
        self.info_label.config(
            text=f"Carpeta: {class_name.upper()} | Imagen {self.current_index + 1}/{len(images)} | "
                 f"{img_path.name} | Bboxes: {bbox_count} ({counts_str}) | "
                 f"Próximo: {next_class_name.upper()} | Total anotadas: {total_annotated} | {annotated}"
        )
    
    def redraw_all_bboxes(self):
        """Redibuja todos los bboxes guardados con colores según su clase"""
        self.canvas.delete('bbox')
        
        for bbox in self.current_bboxes:
            x1 = bbox['x1'] / self.scale_x
            y1 = bbox['y1'] / self.scale_y
            x2 = bbox['x2'] / self.scale_x
            y2 = bbox['y2'] / self.scale_y
            
            color = self.get_class_color(bbox['class'])
            
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=3, tags='bbox'
            )
    
    def on_press(self, event):
        """Inicia el dibujo del rectángulo"""
        self.start_x = event.x
        self.start_y = event.y
        
        if self.rect_id:
            self.canvas.delete(self.rect_id)
    
    def on_drag(self, event):
        """Actualiza el rectángulo mientras se arrastra"""
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
    
    def on_release(self, event):
        """Finaliza el rectángulo y guarda la anotación"""
        if self.start_x is None:
            return
        
        end_x = event.x
        end_y = event.y
        
        x1 = min(self.start_x, end_x)
        x2 = max(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        y2 = max(self.start_y, end_y)
        
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            messagebox.showwarning("Rectángulo muy pequeño", 
                                   "Dibuja un rectángulo más grande")
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
        
        # Convertir a coordenadas originales
        original_x1 = int(x1 * self.scale_x)
        original_y1 = int(y1 * self.scale_y)
        original_x2 = int(x2 * self.scale_x)
        original_y2 = int(y2 * self.scale_y)
        
        # Agregar bbox a la lista CON LA CLASE SELECCIONADA
        new_bbox = {
            'x1': original_x1,
            'y1': original_y1,
            'x2': original_x2,
            'y2': original_y2,
            'width': self.original_width,
            'height': self.original_height,
            'class': self.selected_class_for_bbox  # Usa la clase seleccionada
        }
        
        self.current_bboxes.append(new_bbox)
        
        # Guardar en anotaciones globales
        self.all_annotations[self.current_image_path] = self.current_bboxes.copy()
        
        # Cambiar rectángulo temporal al color de su clase
        self.canvas.delete(self.rect_id)
        self.redraw_all_bboxes()
        
        # Actualizar info
        class_name = CLASSES[self.selected_class_for_bbox]
        images = self.images_by_class[self.current_class]
        remaining = len(images) - self.current_index - 1
        bbox_count = len(self.current_bboxes)
        
        # Contar por clase
        class_counts = {class_id: 0 for class_id in CLASSES.keys()}
        for bbox in self.current_bboxes:
            class_counts[bbox['class']] += 1
        counts_str = " | ".join([f"{CLASSES[cid]}:{cnt}" for cid, cnt in class_counts.items() if cnt > 0])
        
        self.info_label.config(
            text=f"Carpeta: {CLASSES[self.current_class].upper()} | Imagen {self.current_index + 1}/{len(images)} | "
                 f"✅ {class_name.upper()} GUARDADO | Bboxes: {bbox_count} ({counts_str})"
        )
        
        # Guardar automáticamente
        self.save_annotations()
        
        # NO avanzar automáticamente - el usuario decide cuándo seguir
    
    def delete_last_bbox(self):
        """Borra el último bbox agregado"""
        if not self.current_bboxes:
            messagebox.showinfo("Sin bboxes", "No hay bboxes para borrar")
            return
        
        self.current_bboxes.pop()
        
        if self.current_bboxes:
            self.all_annotations[self.current_image_path] = self.current_bboxes.copy()
        else:
            if self.current_image_path in self.all_annotations:
                del self.all_annotations[self.current_image_path]
        
        self.save_annotations()
        self.redraw_all_bboxes()
        
        bbox_count = len(self.current_bboxes)
        class_name = CLASSES[self.current_class]
        images = self.images_by_class[self.current_class]
        
        self.info_label.config(
            text=f"Clase: {class_name.upper()} | Imagen {self.current_index + 1}/{len(images)} | "
                 f"🗑️ Último bbox borrado | Bboxes: {bbox_count}"
        )
    
    def clear_all_annotations(self):
        """Borra todas las anotaciones de la imagen actual"""
        if not self.current_bboxes:
            messagebox.showinfo("Sin anotaciones", "Esta imagen no tiene anotaciones")
            return
        
        if messagebox.askyesno("Confirmar", f"¿Borrar {len(self.current_bboxes)} bbox(es) de esta imagen?"):
            self.current_bboxes = []
            if self.current_image_path in self.all_annotations:
                del self.all_annotations[self.current_image_path]
            self.save_annotations()
            self.redraw_all_bboxes()
            self.load_image()
    
    def next_image(self):
        """Ir a la siguiente imagen"""
        images = self.images_by_class.get(self.current_class, [])
        
        if self.current_index < len(images) - 1:
            self.current_index += 1
            self.load_image()
        else:
            next_class = None
            for class_id in sorted(CLASSES.keys()):
                if class_id > self.current_class and self.images_by_class.get(class_id):
                    next_class = class_id
                    break
            
            if next_class:
                messagebox.showinfo("Clase completa", 
                                  f"Clase '{CLASSES[self.current_class]}' completada.\n"
                                  f"Pasando a '{CLASSES[next_class]}'")
                self.switch_class(next_class)
            else:
                messagebox.showinfo("Fin", "Esta es la última imagen")
    
    def prev_image(self):
        """Ir a la imagen anterior"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()
        else:
            messagebox.showinfo("Inicio", "Esta es la primera imagen de esta clase")
    
    def save_and_exit(self):
        """Guarda y cierra la aplicación"""
        self.save_annotations()
        if self.root:
            self.root.destroy()
    
    def run(self):
        """Inicia la interfaz"""
        if self.root is None:
            return self.all_annotations
        
        self.root.mainloop()
        return self.all_annotations

# ============================================================================
# FUNCIONES DEL PIPELINE
# ============================================================================

def load_existing_annotations():
    """Carga anotaciones previas si existen"""
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'r') as f:
            annotations = json.load(f)
        print(f"✅ Cargadas {len(annotations)} anotaciones previas")
        return annotations
    return {}

def get_images_by_class():
    """Obtiene imágenes organizadas por clase"""
    images_by_class = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif']
    
    for class_id, folder_path in INPUT_FOLDERS.items():
        path = Path(folder_path)
        
        if not path.exists():
            print(f"⚠️  Carpeta no encontrada: {folder_path}")
            continue
        
        images = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
        
        heic_images = [f for f in images if f.suffix.lower() in ['.heic', '.heif']]
        if heic_images and not HEIF_SUPPORT:
            print(f"⚠️  {len(heic_images)} imágenes HEIC en {CLASSES[class_id]} (sin soporte)")
            images = [f for f in images if f.suffix.lower() not in ['.heic', '.heif']]
        
        if images:
            images.sort()
            images_by_class[class_id] = images
            print(f"✅ Clase '{CLASSES[class_id]}': {len(images)} imágenes")
    
    if not images_by_class:
        raise ValueError("❌ No se encontraron imágenes en ninguna carpeta")
    
    return images_by_class

def create_directory_structure():
    """Crea la estructura de carpetas necesaria para YOLO"""
    print("📁 Creando estructura de directorios...")
    
    base_dir = Path(PROJECT_NAME)
    dirs = [
        base_dir / "images" / "train",
        base_dir / "images" / "val",
        base_dir / "labels" / "train",
        base_dir / "labels" / "val"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Estructura creada en {base_dir}/")
    return base_dir

def create_yolo_labels(bboxes, image_width, image_height):
    """Convierte múltiples bboxes a formato YOLO (normalizado)"""
    labels = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        class_id = bbox['class']
        
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return "\n".join(labels) + "\n"

def process_annotations(annotations, base_dir):
    """Procesa las anotaciones y crea el dataset YOLO"""
    print("\n🔄 Procesando anotaciones y creando dataset...")
    
    if not annotations:
        raise ValueError("❌ No hay anotaciones guardadas.")
    
    image_paths = list(annotations.keys())
    import random
    random.shuffle(image_paths)
    
    if len(image_paths) < 10:
        print(f"⚠️  Pocas imágenes ({len(image_paths)}). Usando todas para train y val.")
        train_paths = image_paths
        val_paths = image_paths
    else:
        split_idx = int(len(image_paths) * TRAIN_SPLIT)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
    
    print(f"📊 Train: {len(train_paths)} | Val: {len(val_paths)}")
    
    # Contar objetos por clase
    total_objects = {class_id: 0 for class_id in CLASSES.keys()}
    for bboxes in annotations.values():
        for bbox in bboxes:
            total_objects[bbox['class']] += 1
    
    for class_id, class_name in CLASSES.items():
        print(f"   - {class_name}: {total_objects[class_id]} objetos")
    
    print("🏋️ Procesando conjunto de entrenamiento...")
    for img_path in train_paths:
        bboxes = annotations[img_path]
        src = Path(img_path)
        
        if src.suffix.lower() in ['.heic', '.heif']:
            img = Image.open(src)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            dst_img = base_dir / "images" / "train" / f"{src.stem}.jpg"
            img.save(dst_img, "JPEG", quality=95)
        else:
            dst_img = base_dir / "images" / "train" / src.name
            shutil.copy(src, dst_img)
        
        label_content = create_yolo_labels(bboxes, bboxes[0]['width'], bboxes[0]['height'])
        dst_label = base_dir / "labels" / "train" / f"{src.stem}.txt"
        with open(dst_label, 'w') as f:
            f.write(label_content)
    
    print("✅ Procesando conjunto de validación...")
    for img_path in val_paths:
        bboxes = annotations[img_path]
        src = Path(img_path)
        
        if src.suffix.lower() in ['.heic', '.heif']:
            img = Image.open(src)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            dst_img = base_dir / "images" / "val" / f"{src.stem}.jpg"
            img.save(dst_img, "JPEG", quality=95)
        else:
            dst_img = base_dir / "images" / "val" / src.name
            shutil.copy(src, dst_img)
        
        label_content = create_yolo_labels(bboxes, bboxes[0]['width'], bboxes[0]['height'])
        dst_label = base_dir / "labels" / "val" / f"{src.stem}.txt"
        with open(dst_label, 'w') as f:
            f.write(label_content)
    
    print("✅ Dataset procesado correctamente")
    return len(train_paths), len(val_paths)

def copy_background_images(bg_folder, base_dir, split=0.8):
    """Copia imágenes SIN objetos al dataset YOLO (sin labels)"""
    bg_path = Path(bg_folder)
    if not bg_path.exists():
        print("⚠️  No hay carpeta de background, se omite.")
        return 0, 0

    bg_images = [f for f in bg_path.iterdir() if f.suffix.lower() in [
        '.jpg', '.jpeg', '.png', '.bmp', '.webp'
    ]]

    if not bg_images:
        print("⚠️  No se encontraron imágenes de background.")
        return 0, 0

    import random
    random.shuffle(bg_images)

    split_idx = int(len(bg_images) * split)
    train_bg = bg_images[:split_idx]
    val_bg = bg_images[split_idx:]

    for img in train_bg:
        shutil.copy(img, base_dir / "images" / "train" / img.name)

    for img in val_bg:
        shutil.copy(img, base_dir / "images" / "val" / img.name)

    print(f"🟢 Background agregado → train: {len(train_bg)} | val: {len(val_bg)}")
    return len(train_bg), len(val_bg)


def create_yaml_config(base_dir):
    """Crea el archivo YAML de configuración para YOLO"""
    print("\n📝 Creando archivo de configuración...")
    
    config = {
        'path': str(base_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASSES
    }
    
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuración guardada en {yaml_path}")
    print(f"   Clases: {list(CLASSES.values())}")
    return yaml_path

def train_model(yaml_path, num_images):
    """Entrena el modelo YOLOv11"""
    print("\n🚀 Iniciando entrenamiento YOLOv11...")
    print("=" * 60)
    
    if num_images < 10:
        epochs = 50
    elif num_images < 50:
        epochs = 80
    else:
        epochs = 100
    
    model = YOLO('best.pt')
    
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=960,
        batch=2,
        patience=20,
        save=True,
        project=PROJECT_NAME,
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        device='0',
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
    )
    
    print("\n✅ Entrenamiento completado")
    return model

def validate_model(model, yaml_path):
    """Valida el modelo"""
    print("\n🔍 Validando modelo...")
    metrics = model.val(data=str(yaml_path),
    workers=0
)
    
    print("\n📈 Métricas de validación:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

def export_to_onnx(model, project_name):
    """Exporta el modelo a formato ONNX"""
    print("\n📦 Exportando modelo a formato ONNX...")
    print("=" * 60)
    
    try:
        # Exportar a ONNX
        onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
        
        print(f"✅ Modelo exportado exitosamente a ONNX")
        print(f"📁 Ruta: {onnx_path}")
        
        # También copiar a la carpeta principal del proyecto
        import shutil
        onnx_dest = Path(project_name) / "best.onnx"
        shutil.copy(onnx_path, onnx_dest)
        print(f"📁 Copia guardada en: {onnx_dest}")
        
        return onnx_path
        
    except Exception as e:
        print(f"⚠️  Error al exportar a ONNX: {e}")
        print("   El modelo .pt se ha guardado correctamente")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🎯 PIPELINE YOLOV11 MULTI-CLASE - DETECCIÓN MERCADOLIBRE")
    print("   Clases: LOGO | CONTADOR | CAJA")
    print("   ✨ SOPORTE PARA MÚLTIPLES OBJETOS POR IMAGEN")
    print("=" * 60)
    
    try:
        print(f"\n📸 Buscando imágenes en carpetas...")
        images_by_class = get_images_by_class()
        total_images = sum(len(imgs) for imgs in images_by_class.values())
        print(f"✅ Total: {total_images} imágenes en {len(images_by_class)} clases")
        
        existing_annotations = load_existing_annotations()
        
        print("\n🖱️  Iniciando anotador interactivo multi-bbox...")
        print("   ⚡ Modo inteligente: Solo muestra imágenes SIN anotar")
        print("   ✨ NUEVO: Múltiples clases en la misma imagen")
        print("   Instrucciones:")
        print("   - Las imágenes están organizadas por carpetas (logo/contador/caja)")
        print("   - ANTES de dibujar: Selecciona la CLASE del objeto con los botones")
        print("   - Usa teclas 1=Logo, 2=Contador, 3=Caja para cambiar rápido")
        print("   - Arrastra para dibujar el bbox de esa clase")
        print("   - Puedes mezclar clases: logo + caja, contador + logo, etc.")
        print("   - Los colores muestran la clase: 🟢=Logo 🟡=Contador 🔵=Caja")
        print("   - Clic en 'Siguiente ➡️' cuando termines con la imagen")
        print("   - Usa 'Borrar Último' para eliminar el último bbox")
        
        annotator = MultiClassImageAnnotator(images_by_class, existing_annotations)
        annotations = annotator.run()
        
        if not annotations:
            print("\n❌ No hay anotaciones. El proceso se cancela.")
            return
        
        # Contar total de objetos
        total_objects = 0
        objects_by_class = {class_id: 0 for class_id in CLASSES.keys()}
        
        for bboxes in annotations.values():
            total_objects += len(bboxes)
            for bbox in bboxes:
                objects_by_class[bbox['class']] += 1
        
        print(f"\n✅ Total de imágenes anotadas: {len(annotations)}")
        print(f"✅ Total de objetos detectados: {total_objects}")
        for class_id, class_name in CLASSES.items():
            print(f"   - {class_name}: {objects_by_class[class_id]} objetos")
        
        base_dir = create_directory_structure()
        
        train_count, val_count = process_annotations(annotations, base_dir)
        bg_train, bg_val = copy_background_images(
            "input_files/background",
            base_dir,
            split=TRAIN_SPLIT
        )

        train_count += bg_train
        val_count += bg_val
        yaml_path = create_yaml_config(base_dir)
        
        model = train_model(yaml_path, train_count)
        
        validate_model(model, yaml_path)
        
        # Exportar a ONNX
        onnx_path = export_to_onnx(model, PROJECT_NAME)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"\n📁 Archivos generados:")
        print(f"  - Anotaciones: {ANNOTATIONS_FILE}")
        print(f"  - Dataset: {base_dir}/")
        print(f"  - Modelo PyTorch: {PROJECT_NAME}/train/weights/best.pt")
        if onnx_path:
            print(f"  - Modelo ONNX: {PROJECT_NAME}/best.onnx")
        print(f"\n🎯 Estadísticas finales:")
        print(f"  - Imágenes anotadas: {len(annotations)}")
        print(f"  - Objetos totales: {total_objects}")
        print(f"  - Clases: {list(CLASSES.values())}")
        print(f"\n🚀 Para usar el modelo PyTorch (.pt):")
        print(f"  from ultralytics import YOLO")
        print(f"  model = YOLO('{PROJECT_NAME}/train/weights/best.pt')")
        print(f"  results = model('nueva_imagen.jpg')")
        if onnx_path:
            print(f"\n🚀 Para usar el modelo ONNX (.onnx):")
            print(f"  import onnxruntime as ort")
            print(f"  session = ort.InferenceSession('{PROJECT_NAME}/best.onnx')")
            print(f"  # O con ultralytics:")
            print(f"  model = YOLO('{PROJECT_NAME}/best.onnx')")
        print(f"\n💡 Las clases son: {CLASSES}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    print("📦 Dependencias: pip install ultralytics pillow pyyaml pillow-heif\n")
    main()