import gradio as gr
import pandas as pd
import os
import re # Importar la librería de expresiones regulares
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
from fpdf import FPDF, XPos, YPos
from datetime import datetime
from functools import lru_cache

# --- 1. CONFIGURACIÓN INICIAL Y DE IA ---
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error Crítico: GOOGLE_API_KEY no encontrada.")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)

prompt_template = PromptTemplate.from_template(
    """
    Eres 'EcoRisk AI', un consultor experto en análisis de riesgos ESG.

    **Datos de la Empresa Analizada:**
    - Nombre: {empresa}
    - Riesgo Financiero: {riesgo_f} (Liquidez: {liquidez})
    - Riesgo Ambiental: {riesgo_a} (CO2: {co2} ton, Residuos: {residuos} kg)

    **Benchmark del Sector (Promedios):**
    - Riesgo Financiero Promedio: {avg_riesgo_f_text} (Liquidez Promedio: {avg_liquidez:.2f})
    - Riesgo Ambiental Promedio: {avg_riesgo_a_text} (CO2 Promedio: {avg_co2:.0f} ton, Residuos Promedio: {avg_residuos:.0f} kg)

    **Tu Tarea:**
    Genera un análisis en formato Markdown con las siguientes secciones:

    ### Diagnóstico Comparativo (Benchmark)
    En un párrafo, compara el desempeño de la empresa '{empresa}' contra el promedio del sector. Destaca si está mejor o peor y en qué áreas específicas.

    ### Análisis de Impacto
    En un párrafo, explica cómo los riesgos identificados (financieros y ambientales) pueden estar interconectados y afectar la sostenibilidad del negocio a largo plazo.

    ### Plan de Acción Priorizado
    Una lista numerada con 2-3 recomendaciones accionables y específicas. Justifica cada una en base al diagnóstico comparativo.
    ### Generación de texto
    Devuelve tu respuesta en texto plano sin markdown.
    """
)

chain = prompt_template | llm | StrOutputParser()

# --- 2. LÓGICA DE NEGOCIO Y ANÁLISIS ---
colores = {'BAJO': '#2ECC71', 'MEDIO': '#F1C40F', 'ALTO': '#E74C3C'}
riesgo_map = {'BAJO': 1, 'MEDIO': 2, 'ALTO': 3}
riesgo_map_inv = {v: k for k, v in riesgo_map.items()}

def check_required_columns(df):
    required_cols = ['Empresa', 'Liquidez', 'CO2 (ton)', 'Residuos (kg)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise gr.Error(
            f"Columnas requeridas faltantes: {', '.join(missing_cols)}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

def analizar_riesgo_fila(row):
    liquidez = row.get('Liquidez', 0)
    riesgo_f = 'ALTO' if liquidez < 1 else 'MEDIO' if liquidez < 1.5 else 'BAJO'
    
    co2 = row.get('CO2 (ton)', 0)
    residuos = row.get('Residuos (kg)', 0)
    riesgo_a = 'ALTO' if co2 > 1000 or residuos > 1000 else 'MEDIO' if co2 > 500 or residuos > 500 else 'BAJO'
    
    return riesgo_f, riesgo_a

@lru_cache(maxsize=32)
def invocar_analisis_ia(empresa, liquidez, co2, residuos, riesgo_f, riesgo_a, avg_liquidez, avg_co2, avg_residuos, avg_riesgo_f_text, avg_riesgo_a_text):
    return chain.invoke({
        "empresa": empresa, "liquidez": liquidez, "co2": co2, "residuos": residuos,
        "riesgo_f": riesgo_f, "riesgo_a": riesgo_a, "avg_liquidez": avg_liquidez,
        "avg_co2": avg_co2, "avg_residuos": avg_residuos,
        "avg_riesgo_f_text": avg_riesgo_f_text, "avg_riesgo_a_text": avg_riesgo_a_text
    })

# --- 3. GENERACIÓN DE REPORTE PDF ---

# ✨ NUEVA FUNCIÓN: Limpia el texto Markdown para que sea legible en el PDF
def limpiar_markdown_para_pdf(texto_md):
    # Quitar encabezados (ej. ### Título)
    texto = re.sub(r'###\s+', '', texto_md)
    # Quitar formato de negrita (ej. **texto**)
    texto = re.sub(r'\*\*(.*?)\*\*', r'\1', texto)
    # Quitar formato de lista numerada (ej. 1. Item) y reemplazarlo con un guion
    texto = re.sub(r'^\s*\d+\.\s+', '- ', texto, flags=re.MULTILINE)
    return texto

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Reporte de Diagnóstico EcoRisk', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 12)
        self.multi_cell(0, 10, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln()

def crear_pdf_reporte(empresa, diagnostico_ia, riesgo_f, riesgo_a):
    pdf = PDF()
    pdf.add_page()
    
    pdf.chapter_title(f"Diagnóstico para: {empresa}")
    pdf.chapter_body(f"Nivel de Riesgo Financiero: {riesgo_f}\nNivel de Riesgo Ambiental: {riesgo_a}")
    
    pdf.chapter_title("Análisis y Recomendaciones de IA")
    # ✨ MEJORA: Se usa la nueva función de limpieza
    diagnostico_limpio = limpiar_markdown_para_pdf(diagnostico_ia)
    pdf.chapter_body(diagnostico_limpio)
    
    pdf.set_y(-25)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 10, f'Reporte generado el {datetime.now().strftime("%d/%m/%Y")}', align='C')
    
    filename = f"Reporte_EcoRisk_{empresa.replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# --- 4. FUNCIONES PRINCIPALES DE LA APP ---
def generar_informe_completo(file_obj, empresa_seleccionada):
    if not file_obj or not empresa_seleccionada:
        return "", "", "", "", None, gr.Column(visible=False)

    df = pd.read_excel(file_obj.name)
    check_required_columns(df)

    df[['riesgo_f', 'riesgo_a']] = df.apply(lambda row: pd.Series(analizar_riesgo_fila(row)), axis=1)
    df['riesgo_f_num'] = df['riesgo_f'].map(riesgo_map)
    df['riesgo_a_num'] = df['riesgo_a'].map(riesgo_map)
    
    avg_liquidez = df['Liquidez'].mean()
    avg_co2 = df['CO2 (ton)'].mean()
    avg_residuos = df['Residuos (kg)'].mean()
    avg_riesgo_f_text = riesgo_map_inv[round(df['riesgo_f_num'].mean())]
    avg_riesgo_a_text = riesgo_map_inv[round(df['riesgo_a_num'].mean())]

    datos_empresa = df[df['Empresa'] == empresa_seleccionada].iloc[0]
    riesgo_f, riesgo_a = datos_empresa['riesgo_f'], datos_empresa['riesgo_a']
    
    analisis_ia = invocar_analisis_ia(
        empresa_seleccionada, datos_empresa['Liquidez'], datos_empresa['CO2 (ton)'], datos_empresa['Residuos (kg)'],
        riesgo_f, riesgo_a, avg_liquidez, avg_co2, avg_residuos, avg_riesgo_f_text, avg_riesgo_a_text
    )

    out_riesgo_f_html = f"""<div style='background-color:{colores[riesgo_f]}; padding: 15px; border-radius: 10px; text-align: center; color: white;'><h4>Riesgo Financiero</h4><p style='font-size: 22px; font-weight: bold; margin:0;'>{riesgo_f}</p></div>"""
    out_riesgo_a_html = f"""<div style='background-color:{colores[riesgo_a]}; padding: 15px; border-radius: 10px; text-align: center; color: white;'><h4>Riesgo Ambiental</h4><p style='font-size: 22px; font-weight: bold; margin:0;'>{riesgo_a}</p></div>"""
    
    benchmark_html = f"""<div style='border: 2px solid #E5E7EB; padding: 15px; border-radius: 10px;'><h4 style='text-align:center; margin-top:0;'>Benchmark del Sector (Promedio)</h4><p><strong>Financiero:</strong> {avg_riesgo_f_text} (Liquidez: {avg_liquidez:.2f})</p><p><strong>Ambiental:</strong> {avg_riesgo_a_text} (CO2: {avg_co2:.0f}t, Residuos: {avg_residuos:.0f}kg)</p></div>"""
    
    pdf_path = crear_pdf_reporte(empresa_seleccionada, analisis_ia, riesgo_f, riesgo_a)
    
    return out_riesgo_f_html, out_riesgo_a_html, benchmark_html, analisis_ia, pdf_path, gr.Column(visible=True)

def actualizar_dropdown_empresas(file_obj):
    if not file_obj:
        return gr.Dropdown(choices=[], interactive=False)
    
    df = pd.read_excel(file_obj.name)
    check_required_columns(df)
    empresas = df['Empresa'].dropna().unique().tolist()
    return gr.Dropdown(choices=empresas, value=empresas[0] if empresas else None, interactive=True)

# --- 5. INTERFAZ DE USUARIO CON GRADIO ---
css = """
#cont_informe {border-radius: 20px;}
#logo_dux {max-width: 250px; margin: auto; background-color: transparent;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="gray"), css=css) as app:
    gr.Image("logo_dux.jpg", elem_id="logo_dux", show_label=False, show_download_button=False, container=False)
    gr.Markdown("<h1 style='text-align: center;'>Bienvenido a EcoRisk 🌿 v2.0</h1><h3 style='text-align: center; color: #4B5563;'>Tu consultor IA para un futuro sostenible y rentable</h3>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### **1.** Carga tus Datos")
            file_uploader = gr.File(label="Sube tu archivo Excel", file_types=['.xlsx'], height=130)
            
            gr.Markdown("### **2.** Elige la Empresa")
            empresa_selector = gr.Dropdown(label="Empresa a Analizar", interactive=False)
            
            btn_analizar = gr.Button("Generar Informe Avanzado", variant="primary")

        with gr.Column(scale=3):
            with gr.Column(visible=False, elem_id="cont_informe") as cont_informe:
                with gr.Tabs() as tabs:
                    with gr.TabItem("Dashboard de Riesgos", id=0):
                        with gr.Row():
                            out_riesgo_f = gr.Markdown()
                            out_riesgo_a = gr.Markdown()
                        gr.Markdown("---")
                        out_benchmark = gr.Markdown()
                    
                    with gr.TabItem("Análisis Detallado (IA)", id=1):
                        out_analisis_ia = gr.Markdown()
                    
                    with gr.TabItem("Descargar Reporte", id=2):
                        out_pdf = gr.File(label="Descarga tu reporte en PDF")

    file_uploader.upload(fn=actualizar_dropdown_empresas, inputs=file_uploader, outputs=empresa_selector)
    btn_analizar.click(
        fn=generar_informe_completo,
        inputs=[file_uploader, empresa_selector],
        outputs=[out_riesgo_f, out_riesgo_a, out_benchmark, out_analisis_ia, out_pdf, cont_informe]
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True)