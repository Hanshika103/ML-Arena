import streamlit as st


# ==========================================
# Hero Section
# ==========================================

def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# Section Heading
# ==========================================

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")

    if subtitle:
        st.caption(subtitle)


# ==========================================
# Feature Card
# ==========================================

def feature_card(icon: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="card">
            <h3>{icon} {title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# Metric Card
# ==========================================

def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="card" style="text-align:center;">
            <p style="font-size:15px;
                      color:#6B7280;
                      margin-bottom:8px;">
                {title}
            </p>

            <h2 style="
                color:#C89B3C;
                margin:0;
                font-size:34px;
            ">
                {value}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# Information Box
# ==========================================

def info_box(message: str):
    st.info(message)


# ==========================================
# Success Box
# ==========================================

def success_box(message: str):
    st.success(message)


# ==========================================
# Warning Box
# ==========================================

def warning_box(message: str):
    st.warning(message)


# ==========================================
# Footer
# ==========================================

def page_footer():

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center;
                    color:#6B7280;
                    padding:10px;">
            ML Arena • Intelligent Machine Learning Benchmark Platform <br>
            Version 1.0
        </div>
        """,
        unsafe_allow_html=True,
    )