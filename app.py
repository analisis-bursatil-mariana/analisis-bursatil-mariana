# app.py
# Mi primer app de CV con Streamlit (carga la foto del archivo IMG_8781.PNG)

import os
from datetime import date
import streamlit as st

st.set_page_config(
    page_title="CV - Mariana Hernández Arroyo",
    page_icon="📄",
    layout="wide"
)

# ---- Foto (desde archivo local) ----
FOTO = "IMG_8781.PNG"  # ya la tienes en la carpeta

col_left, col_right = st.columns([3, 1], vertical_alignment="center")
with col_left:
    st.title("📄 Curriculum Vitae")
    st.subheader("Mariana Hernández Arroyo")
    st.caption("Lic. en Administración y Finanzas — 9º semestre")

with col_right:
    if os.path.exists(FOTO):
        st.image(FOTO, caption="Mariana Hernández", width=180)
    else:
        st.markdown(
            """
            <div style="width:180px;height:180px;border:2px dashed #bbb;border-radius:16px;
                        display:flex;align-items:center;justify-content:center;opacity:0.8;padding:10px;">
              <span style="font-size:13px;text-align:center;">
                (No se encontró <b>IMG_8781.PNG</b>)<br>Coloca la foto junto a <b>app.py</b>.
              </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---- Sidebar: contacto ----
with st.sidebar:
    st.header("Contacto")
    st.write("📍 **Ubicación:** Zapopan, Jalisco, 45010")
    st.write("📞 **Teléfono:** 669 107 0921")
    st.write("✉️ **Email:** marianaharroyo25@gmail.com")
    st.write("📸 **Instagram:** @marianaa_ha")
    st.write("🎂 **Edad:** 20 años")

    st.divider()
    st.header("Idiomas")
    st.write("- Inglés — **Avanzado**")
    st.write("- Alemán — **Básico**")

    st.divider()
    st.header("Habilidades")
    for h in [
        "Gestión personal",
        "Capacidad negociadora",
        "Toma de decisiones",
        "Determinación para resultados",
        "Adaptación al cambio",
        "Liderazgo",
        "Organización de actividades",
    ]:
        st.write(f"• {h}")

    st.divider()
    st.header("Herramientas")
    st.write("Excel · Word · PowerPoint · Canva")

# ---- Sobre mí ----
st.header("Sobre mí")
st.write(
    "Estudiante universitaria comprometida con su crecimiento personal y profesional, "
    "muy trabajadora y responsable. Busco colaborar para emplear mis competencias en "
    "el logro de objetivos y aportar valor al equipo."
)

# ---- Educación ----
st.header("Educación")
c1, c2 = st.columns([1, 3])
with c1: st.write("**2021 – a la fecha**")
with c2: st.write("**Universidad Panamericana** — Licenciatura en Administración y Finanzas")

c1, c2 = st.columns([1, 3])
with c1: st.write("**2005 – 2021**")
with c2: st.write("**Colegio Andes de Mazatlán** — Preescolar a Preparatoria")

# ---- Experiencia ----
st.header("Experiencia profesional")
c1, c2 = st.columns([1, 3])
with c1: st.write("**2024 – Actualidad**")
with c2:
    st.write("**BlackBull Markets** — *Operations Officer*")
    st.write("- Gestión operativa y soporte a procesos internos.\n"
             "- Colaboración transversal para asegurar continuidad operativa.")

c1, c2 = st.columns([1, 3])
with c1: st.write("**2023 – 2024**")
with c2:
    st.write("**Universidad Panamericana** — *Becaria*")
    st.write("- Becaria en la Carrera de Administración y Finanzas.\n"
             "- Becaria en el Departamento de Arte y Cultura.")

c1, c2 = st.columns([1, 3])
with c1: st.write("**2023 – Actualidad**")
with c2:
    st.write("**IMEF Universitario** — *Presidente MDL UP GDL*")
    st.write("- Liderazgo de capítulo; organización de actividades académicas y de vinculación.")

c1, c2 = st.columns([1, 3])
with c1: st.write("**2022 – 2023**")
with c2:
    st.write("**IMEF Universitario** — *Vicepresidente MDL UP GDL*")
    st.write("- Coordinación de miembros y proyectos de desarrollo profesional.")

c1, c2 = st.columns([1, 3])
with c1: st.write("**2020 – 2023**")
with c2:
    st.write("**Little Bazar** — *Emprendimiento propio*")
    st.write("- Diseño, elaboración y venta de accesorios.\n"
             "- Creación de marca y administración de redes.")

# ---- Extra ----
st.header("Información relevante")
st.write("- Tesorera de Trasciende en Colegio Andes (2018–2021).")
st.write("- Labor social en Amigos de San José María E.A.C.")
st.write("- Miembro Club Net (2010–2019).")

st.header("Hobbies")
st.write("- Hacer voluntariado · Pintar · Cocinar · Viajar")

st.divider()
st.caption(f"Última actualización: {date.today().strftime('%d/%m/%Y')}")
st.caption("App hecha con ♥ en Streamlit por Mariana (mi primera app).")


