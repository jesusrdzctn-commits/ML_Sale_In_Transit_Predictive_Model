"""
Aplicación GUI Robusta para Modelos Predictivos
Diseñada para replicar exactamente el comportamiento de los sistemas originales
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from datetime import datetime
from advanced_processor import AdvancedProcessor

class PredictiveModelGUI:
    """Interfaz gráfica principal"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Tránsitos")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_path = tk.StringVar()
        self.prediction = tk.StringVar(value="doble")
        self.monitoring = tk.StringVar()
        
        # Configurar estilo
        self.setup_style()
        
        # Crear interfaz
        self.create_widgets()
        
        # Centrar ventana
        self.center_window()
    
    def setup_style(self):
        """Configurar estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilos personalizados
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 9), foreground='gray')
        style.configure('Process.TButton', font=('Arial', 10, 'bold'))
    
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título principal
        title_label = ttk.Label(main_frame, text="Sistema de Predicción de Tránsitos", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 30))
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="20")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        config_frame.columnconfigure(1, weight=1)
        
        # Archivo de datos
        ttk.Label(config_frame, text="Archivo de Datos:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        input_frame = ttk.Frame(config_frame)
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        input_entry = ttk.Entry(input_frame, textvariable=self.input_file, state='readonly')
        input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(input_frame, text="Examinar", command=self.browse_input_file).grid(row=0, column=1)
        
        # Mes de monitoreo
        ttk.Label(config_frame, text="Mes de Monitoreo:", style='Subtitle.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        month_frame = ttk.Frame(config_frame)
        month_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        month_entry = ttk.Entry(month_frame, textvariable=self.monitoring, width=15)
        month_entry.grid(row=0, column=0, padx=(0, 10))
        ttk.Label(month_frame, text="(ej: sep-25, oct-25)", style='Info.TLabel').grid(row=0, column=1, sticky=tk.W)
        
        # Tipo de predicción
        ttk.Label(config_frame, text="Tipo de Predicción:", style='Subtitle.TLabel').grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        pred_frame = ttk.Frame(config_frame)
        pred_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        ttk.Radiobutton(pred_frame, text="Predicción Doble", variable=self.prediction, value="doble").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Radiobutton(pred_frame, text="Predicción Simple", variable=self.prediction, value="simple").grid(row=0, column=1, sticky=tk.W)
        
        # Carpeta de salida
        ttk.Label(config_frame, text="Carpeta de Salida:", style='Subtitle.TLabel').grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        output_frame = ttk.Frame(config_frame)
        output_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        output_frame.columnconfigure(0, weight=1)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, state='readonly')
        output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(output_frame, text="Examinar", command=self.browse_output_folder).grid(row=0, column=1)
        
        # Botón de procesamiento
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        self.process_button = ttk.Button(process_frame, text="🚀 Iniciar Procesamiento", 
                                       command=self.run_processing, style='Process.TButton')
        self.process_button.pack()
        
        # Log de progreso
        log_frame = ttk.LabelFrame(main_frame, text="Progreso", padding="15")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Text widget con scrollbar
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Footer con branding
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        ttk.Label(footer_frame, text="Control Efficiency", style='Info.TLabel').pack()
    
    def center_window(self):
        """Centrar la ventana en la pantalla"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def browse_input_file(self):
        """Seleccionar archivo de datos de entrada"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
    
    def browse_output_folder(self):
        """Seleccionar carpeta de salida"""
        folder = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if folder:
            self.output_path.set(folder)
    
    def validate_inputs(self):
        """Validar entradas del usuario"""
        if not self.input_file.get():
            messagebox.showerror("Error", "Por favor selecciona un archivo de datos")
            return False
        
        if not self.monitoring.get():
            messagebox.showerror("Error", "Por favor ingresa el mes de monitoreo")
            return False
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Por favor selecciona una carpeta de salida")
            return False
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "El archivo de datos no existe")
            return False
        
        return True
    
    def log_message(self, message):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_processing(self):
        """Ejecutar el procesamiento en un hilo separado"""
        if not self.validate_inputs():
            return
        
        # Deshabilitar botón mientras se procesa
        self.process_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)
        
        # Iniciar procesamiento en un hilo separado
        thread = threading.Thread(target=self.process_predictions)
        thread.start()
    
    def process_predictions(self):
        """Procesar las predicciones (ejecutado en hilo separado) - Solución Simple y Funcional"""
        try:
            input_file = self.input_file.get()
            monitoring = self.monitoring.get()
            prediction = self.prediction.get()
            output_dir = self.output_path.get()
            
            self.log_message("🚀 Iniciando procesamiento...")
            self.log_message("="*50)
            self.log_message("INICIO DEL PROCESO")
            self.log_message("="*50)
            self.log_message(f"Leyendo base de datos {input_file}")
            self.log_message(f"Monitoreo {monitoring}")
            self.log_message(f"Tipo de predicción: {prediction}")
            
            # Crear carpeta
            if prediction == "doble":
                folder = os.path.join(output_dir, f"Predicciones Dobles {monitoring}")
            else:
                folder = os.path.join(output_dir, f"Predicciones Simples {monitoring}")
            
            check = os.path.isdir(folder)
            if check:
                self.log_message(f"✓ Carpeta existente: {folder}")
            else:
                os.makedirs(folder)
                self.log_message(f"✓ Carpeta creada: {folder}")
            
            self.log_message(f"📁 Carpeta de destino: {folder}")
            self.log_message(f"📂 Ruta absoluta: {os.path.abspath(folder)}")
            
            # Crear procesador avanzado
            processor = AdvancedProcessor(self.log_message)
            
            # Procesar datos
            input_data = processor.clean_data(input_file, monitoring)
            
            if input_data.empty:
                self.log_message("❌ ERROR: No se encontraron datos para el mes de monitoreo especificado.")
                messagebox.showerror("Error", "No se encontraron datos para el mes de monitoreo especificado.")
                return
            
            # Entrenar modelos avanzados con mayor peso a los últimos 12 meses
            self.log_message("\n" + "="*50)
            self.log_message("ENTRENANDO MODELOS AVANZADOS")
            self.log_message("="*50)
            
            if not processor.train_all_models(input_data):
                self.log_message("❌ ERROR: No se pudieron entrenar los modelos.")
                messagebox.showerror("Error", "No se pudieron entrenar los modelos.")
                return
            
            mixins = input_data["Mixing Nombre"].unique()
            self.log_message(f"🔍 Mixings encontrados: {len(mixins)}")
            self.log_message(f"📋 Lista de mixings: {list(mixins)}")
            
            files_created = 0
            for i, mixin in enumerate(mixins, 1):
                self.log_message(f"\n--- Procesando Mixing {i}/{len(mixins)} ---")
                self.log_message(f"🎯 Generando Predicciones de SMC: {mixin}")
                mixin_data = input_data[input_data["Mixing Nombre"] == mixin]
                self.log_message(f"📊 Datos para {mixin}: {len(mixin_data)} registros")
                
                try:
                    success = processor.process_mixing(mixin, mixin_data, folder, prediction)
                    if success:
                        files_created += 1
                        self.log_message(f"✅ Predicciones generadas para {mixin}")
                    else:
                        self.log_message(f"❌ Error procesando {mixin}")
                        
                except Exception as e:
                    self.log_message(f"❌ ERROR procesando {mixin}: {str(e)}")
                    self.log_message(f"🔍 Tipo de error: {type(e).__name__}")
            
            self.log_message(f"\n📈 Resumen de procesamiento:")
            self.log_message(f"   - Mixings procesados exitosamente: {files_created}/{len(mixins)}")
            
            # Verificar archivos creados
            self.log_message(f"\n🔍 Verificando archivos en carpeta antes de unir:")
            if os.path.exists(folder):
                files_in_folder = os.listdir(folder)
                csv_files = [f for f in files_in_folder if f.endswith('.csv')]
                self.log_message(f"   - Total archivos en carpeta: {len(files_in_folder)}")
                self.log_message(f"   - Archivos CSV encontrados: {len(csv_files)}")
                if csv_files:
                    self.log_message(f"   - Archivos CSV: {csv_files}")
                else:
                    self.log_message(f"   - ⚠️  No hay archivos CSV en la carpeta!")
            else:
                self.log_message(f"   - ❌ La carpeta {folder} no existe!")
            
            # Unir datos
            self.log_message("\n" + "="*50)
            self.log_message("UNIENDO ARCHIVOS")
            self.log_message("="*50)
            self.log_message("Creando archivo final")
            processor.join_data(folder, prediction, monitoring)
            self.log_message("✅ Proceso terminado exitosamente.")
            
            messagebox.showinfo("Éxito", "Predicciones generadas exitosamente!")
            
        except Exception as e:
            self.log_message(f"❌ Error general: {str(e)}")
            messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento: {str(e)}")
        finally:
            self.process_button.config(state=tk.NORMAL)

def main():
    """Función principal"""
    root = tk.Tk()
    app = PredictiveModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
