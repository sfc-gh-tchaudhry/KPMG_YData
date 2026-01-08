"""YData Synthetic Data Generator - Streamlit App"""
from faulthandler import disable

import streamlit as st
import pandas as pd
import os
import tempfile
from io import BytesIO
import base64
from pathlib import Path

os.environ['YDATA_LICENSE_KEY'] = '74ff0c2a-ae55-41ba-bb00-976bee030b68'

from ydata.connectors import LocalConnector
from ydata.dataset.filetype import FileType
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.synthesizers.regular.model import RegularSynthesizer
from ydata.profiling import ProfileReport

from streamlit.components import v1 as components

# Page config
st.set_page_config(
    page_title="YData Synthetic Data Generator",
    page_icon="🧬",
    layout="wide"
)

#Init session state variables
st.session_state.setdefault("real_data_path", None)
st.session_state.setdefault("metadata_path", None)
st.session_state.setdefault("synthetic_data", None)

st.session_state.setdefault("profile_compare", None)
st.session_state.setdefault("quality_report", None)
st.session_state.setdefault("quality_report_bytes", None)

@st.cache_data
def load_data(path: str) -> Dataset:
    connector = LocalConnector()
    data = connector.read_file(path=path, file_type=FileType.CSV)
    st.session_state.real_data_path = path
    return data

@st.cache_data
def load_metadata(path: str) -> Metadata:
    metadata = Metadata.load(path)
    return metadata

def generate_compare_profiling(data, synth_data):
    """Generate profile report comparing"""
    with st.spinner("Generating compare profiling..."):
        real_profile = ProfileReport(data, title='Original dataset')
        synth_profile = ProfileReport(synth_data, title='Synthetic dataset')

        compare = real_profile.compare(synth_profile)
        compare.config.html.navbar_show = False
        st.session_state.profile_compare = compare.to_html()

@st.dialog("Dataset Compare Profiling report", width="large")
def show_profile():
    st.html(st.session_state.profile_compare)

@st.dialog("Synthetic Data Quality Report", width='large')
def show_quality_report():
    path_pdf = Path(st.session_state.quality_report)
    pdf_bytes = path_pdf.read_bytes()

    st.pdf(pdf_bytes, height=800)

def generate_pdf_quality_report(data, synth_data, metadata, target_col):
    """Generate the PDF quality report"""
    from ydata.report import SyntheticDataProfile

    if target_col=='None':
        target_col = None

    with st.spinner("Calculating quality report..."):
        quality_report = SyntheticDataProfile(
            real=data,
            synth=synth_data,
            metadata=metadata,
            target=target_col
        )

        metrics = quality_report._report_info['info_metrics']

        st.markdown(metrics)

        quality_report.generate_report(output_path='synthetic_data_quality_report.pdf')

        st.session_state.quality_report='synthetic_data_quality_report.pdf'

    return metrics

# Custom styling - Light theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 50%, #dfe6ed 100%);
    }
    .main-header {
        font-family: 'Courier New', monospace;
        color: #2563eb;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        color: #dc2626;
        font-weight: 600;
    }
    .success-box {
        background: rgba(37, 99, 235, 0.1);
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #dc2626, #2563eb);
        color: white;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(220, 38, 38, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧬 YData Synthetic Data Generator</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None

# File upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h3 class="sub-header">📁 Upload Your Data</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload the CSV file you want to use for training the synthetic data model"
    )

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Data Preview</h3>', unsafe_allow_html=True)
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
    
    # Show data preview
    st.markdown("### Data Preview (First 10 rows)")
    st.dataframe(df.head(10), width='stretch')
    
    # Show column info
    st.markdown("---")
    st.markdown('<h3 class="sub-header">⚙️ Model Configuration</h3>', unsafe_allow_html=True)
    
    config_col1, config_col2, config_col3, config_col4= st.columns(4)
    
    with config_col1:
        # Column selection for conditioning
        columns_list = ["None (No conditioning)"] + list(df.columns)
        selected_column = st.selectbox(
            "🎯 Select column to condition on",
            options=columns_list,
            help="The model will learn patterns conditioned on this column. Useful for classification targets or important categorical variables."
        )

    with config_col2:
        # Balancing option
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        use_balancing = st.checkbox(
            "⚖️ Balance output",
            value=False,
            help="If enabled, synthetic data will have balanced classes for the conditioned column"
        )

    with config_col3:
        privacy_level = st.selectbox(
            "Privacy Level",
            options=['High Fidelity', 'Balanced', 'High Privacy'],
            help="Model controls powered by different privacy that enables different privacy levels and control over synthetic data quality's utility and privacy."
        )

    with config_col4:
        target_col = st.selectbox(
            "Target Column",
            options=['None'] +list(df.columns),
            help="Target variable will enable more validations in terms of the quality of the synthetic data generated."
        )

    samples, train_button = st.columns([2, 1])

    with samples:
        # Number of samples to generate
        n_samples = st.number_input(
            "📈 Number of samples to generate",
            min_value=10,
            max_value=100000,
            value=len(df),
            step=100,
            help="How many synthetic rows to generate"
        )

    with train_button:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Train Model & Generate Synthetic Data", width='stretch'):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        uploaded_file.seek(0)
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    data = load_data(tmp_path)

                    # Create synthesizer and metadata
                    synth = RegularSynthesizer()
                    metadata = Metadata(dataset=data)

                    # Save the metadata
                    metadata.save('metadata.pkl')
                    st.session_state.metadata_path = 'metadata.pkl'

                    # Fit the model
                    if selected_column == "None (No conditioning)":
                        synth.fit(X=data, metadata=metadata)
                    else:
                        synth.fit(X=data, metadata=metadata, condition_on=selected_column)

                    st.session_state.trained_model = synth

                    # Generate synthetic samples
                    if selected_column == "None (No conditioning)":
                        synth_sample = synth.sample(n_samples=n_samples)
                    else:
                        synth_sample = synth.sample(n_samples=n_samples, balancing=use_balancing)

                    st.session_state.synthetic_data = synth_sample

                    ## Calculate the profiling compare
                    generate_compare_profiling(data=data,
                                               synth_data=synth_sample)

                    ## Calculate PDF report quality metrics
                    generate_pdf_quality_report(data=data,
                                                synth_data=synth_sample,
                                                metadata=metadata,
                                                target_col=target_col)

                    # Cleanup temp file
                    os.unlink(tmp_path)

                    st.success(f"✅ Successfully generated {len(synth_sample):,} synthetic records!")

                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    raise e

    
    # Display results if available
    if st.session_state.synthetic_data is not None:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎉 Evaluate Synthetic Data</h2>', unsafe_allow_html=True)

        synth_data = st.session_state.synthetic_data

        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            with st.container(border=True):
                st.markdown("<h5 style='text-align: center; color: black;'>Download synthetic data</h5>", unsafe_allow_html=True)
                st.write(f"Download the generated synthetic dataset with **{len(df):,}** rows as a CSV file.")

                if st.session_state.synthetic_data is not None:
                    # Download button
                    csv_buffer = BytesIO()

                    synth_data.to_pandas().to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Synthetic Data (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="synthetic_data.csv",
                        mime="text/csv",
                        width='stretch'
                    )

        with c2:
            with st.container(border=True):
                st.markdown("<h5 style='text-align: center; color: black;'>Compare profiling</h5>", unsafe_allow_html=True)
                st.write(f"Statistical comparison between original and synthetic data. Generated on-demand.")

                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📥 Download Compare",
                        data=st.session_state.profile_compare,
                        file_name="report.html",
                        mime="text/html",
                        width='stretch',
                        disabled = st.session_state.profile_compare is None

                    )

                with col2:
                    if st.button("Visualize compare profiling", width='stretch'):
                        if st.session_state.profile_compare is not None:
                            show_profile()
                        else:
                            st.write("Please generate the compare profile first.")

        with c3:
            with st.container(border=True):
                st.markdown("<h5 style='text-align: center; color: black;'>Quality report</h5>", unsafe_allow_html=True)
                st.write('Detailed quality metrics and evaluation of the synthetic data. Generated on-demand.')

                download, visualize = st.columns(2)

                with download:
                    pdf_path = Path(st.session_state.quality_report)

                    st.download_button(
                        label="📥 Download Quality Report",
                        data=pdf_path.read_bytes(),
                        file_name="quality_report.pdf",
                        mime="text/html",
                        width='stretch',
                        disabled=st.session_state.quality_report is None

                    )

                with visualize:
                    if st.button("Visualize quality report", width='stretch'):
                        if st.session_state.quality_report is not None:
                            show_quality_report()
                        else:
                            st.write("Please generate the quality report first.")

        st.markdown("### Synthetic data preview")

        # Preview synthetic data
        st.markdown("**Synthetic Data Preview (First 10 rows):**")
        synth_df = st.session_state.synthetic_data.to_pandas()
        st.dataframe(synth_df.head(10), width='stretch')

        # Stats comparison
        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:
            st.markdown("**Original Data Stats:**")
            st.dataframe(df.describe(), width='stretch')

        with stat_col2:
            st.markdown("**Synthetic Data Stats:**")
            st.dataframe(synth_df.describe(), width='stretch')

        # Option to save model
        st.markdown("---")
        st.markdown('<h3 class="sub-header">💾 Save Model</h3>', unsafe_allow_html=True)

        model_name = st.text_input("Model filename", value="synth_model.pkl")
        if st.button("Save Model to Disk"):
            try:
                st.session_state.trained_model.save(path=f"./{model_name}")
                st.success(f"✅ Model saved as {model_name}")
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload a CSV file to get started!")
    
    st.markdown("""
    ### How to use:
    1. **Upload** your CSV data file
    2. **Select** a column to condition the model on (optional)
    3. **Configure** the number of synthetic samples to generate
    4. **Click** "Train Model & Generate" to create synthetic data
    5. **Download** the generated synthetic data
    
    ### About YData Synthesizer:
    The RegularSynthesizer learns the statistical patterns and correlations 
    in your data, then generates new synthetic records that preserve these 
    characteristics while protecting privacy.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Powered by YData SDK</p>",
    unsafe_allow_html=True
)

