"""YData Synthetic Data Generator - Streamlit App"""

import streamlit as st
import pandas as pd
import os
import tempfile
from io import BytesIO

os.environ['YDATA_LICENSE_KEY'] = '74ff0c2a-ae55-41ba-bb00-976bee030b68'

from ydata.connectors import LocalConnector
from ydata.dataset.filetype import FileType
from ydata.metadata import Metadata
from ydata.synthesizers.regular.model import RegularSynthesizer

# Page config
st.set_page_config(
    page_title="YData Synthetic Data Generator",
    page_icon="🧬",
    layout="wide"
)

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
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        # Column selection for conditioning
        columns_list = ["None (No conditioning)"] + list(df.columns)
        selected_column = st.selectbox(
            "🎯 Select column to condition on",
            options=columns_list,
            help="The model will learn patterns conditioned on this column. Useful for classification targets or important categorical variables."
        )
    
    with config_col2:
        # Number of samples to generate
        n_samples = st.number_input(
            "📈 Number of samples to generate",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
            help="How many synthetic rows to generate"
        )
    
    with config_col3:
        # Balancing option
        use_balancing = st.checkbox(
            "⚖️ Balance output",
            value=True,
            help="If enabled, synthetic data will have balanced classes for the conditioned column"
        )
    
    st.markdown("---")
    
    # Train and generate section
    col_train, col_download = st.columns([1, 1])
    
    with col_train:
        if st.button("🚀 Train Model & Generate Synthetic Data", width='stretch'):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        uploaded_file.seek(0)
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Initialize connector and read data
                    connector = LocalConnector()
                    data = connector.read_file(path=tmp_path, file_type=FileType.CSV)
                    
                    # Create synthesizer and metadata
                    synth = RegularSynthesizer()
                    metadata = Metadata(dataset=data)
                    
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
                    
                    # Convert to pandas
                    synth_df = synth_sample.to_pandas()
                    st.session_state.synthetic_data = synth_df
                    
                    # Cleanup temp file
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Successfully generated {len(synth_df):,} synthetic records!")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    raise e
    
    # Display results if available
    if st.session_state.synthetic_data is not None:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🎉 Generated Synthetic Data</h3>', unsafe_allow_html=True)
        
        synth_df = st.session_state.synthetic_data
        
        # Stats comparison
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown("**Original Data Stats:**")
            st.dataframe(df.describe(), width='stretch')
        
        with stat_col2:
            st.markdown("**Synthetic Data Stats:**")
            st.dataframe(synth_df.describe(), width='stretch')
        
        # Preview synthetic data
        st.markdown("### Synthetic Data Preview (First 10 rows)")
        st.dataframe(synth_df.head(10), width='stretch')
        
        # Download button
        csv_buffer = BytesIO()
        synth_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Synthetic Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="synthetic_data.csv",
            mime="text/csv",
            width='stretch'
        )
        
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

