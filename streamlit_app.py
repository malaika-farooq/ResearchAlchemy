import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="Research Alchemy",
    page_icon="🔮",
    layout="wide",
)

# Access API key from Streamlit secrets and set up the base URL
aiml_api_key = st.secrets["aiml_api_key"]
base_url = "https://api.aimlapi.com/v1"  # Update this if the base URL is different

# Initialize OpenAI client with the API key and base URL
client = OpenAI(
    api_key=aiml_api_key,
    base_url=base_url
)

# Add custom CSS to style headers and buttons
def local_css():
    st.markdown(f"""
        <style>
            /* Header colors */
            h1, h2, h3, h4, h5, h6 {{
                color: #f54e25;
            }}
            /* Button styles */
            .stButton > button {{
                background-color: #6044ea;
                color: #FFFFFF;
            }}
            /* Tab styles */
            .stTabs [role="tablist"] .css-1hynsf2 {{
                background-color: #f54e25;
            }}
            .stTabs [role="tablist"] .css-1hynsf2 [data-baseweb="tab"] {{
                color: #FFFFFF;
            }}
            /* Sidebar styles */
            .css-1d391kg {{
                background-color: #f54e25;
            }}
            /* Main content background */
            .css-18e3th9 {{
                background-color: #FFFFFF;
            }}
        </style>
        """, unsafe_allow_html=True)

local_css()

# Add a logo and app info in the sidebar
with st.sidebar:
    st.image("assets/research-alchemy.png", use_column_width=False)
    st.title("Research Alchemy")
    st.write("""
    Welcome to **Research Alchemy**!

    This app is made for researchers and scientists to help them with:

    1. **Data Preprocessing & Cleaning Tool**
    2. **Hypothesis Generation Tool**
    3. **AI-Powered Research Dashboard**

    Use the tabs to navigate through each project and explore the details.
    """)

# Create three tabs
tab1, tab2, tab3 = st.tabs([
    "1. Data Preprocessing & Cleaning Tool",
    "2. Hypothesis Generation Tool",
    "3. AI-Powered Research Dashboard"
])

with tab1:
    st.header("1. Scientific Data Preprocessing & Cleaning Platform")
    st.subheader("Description")
    st.write("""
    A platform that focuses on automating the often tedious task of data cleaning, normalizing, preprocessing, and feature selection for scientific datasets. It can detect outliers, suggest transformations, and automate feature engineering.
    """)

    st.subheader("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the dataset
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Data Preprocessing
        st.subheader("🔧 Data Preprocessing")

        # Handle missing values
        if st.checkbox("Fill missing values with mean"):
            df.fillna(df.mean(), inplace=True)
            st.write("Filled missing values with column means.")

        # Select columns for analysis
        st.write("### Select Columns for Analysis")
        columns = st.multiselect(
            "Select numerical columns for analysis",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            default=df.select_dtypes(include=[np.number]).columns.tolist()
        )

        if columns:
            data = df[columns]
            # Anomaly Detection
            st.subheader("🚨 Anomaly Detection using Isolation Forest")
            contamination = st.slider(
                "Select contamination level (proportion of outliers)",
                min_value=0.01, max_value=0.5, value=0.1, step=0.01
            )
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            preds = iso_forest.fit_predict(data)
            df['Anomaly'] = preds
            outliers = df[df['Anomaly'] == -1]
            st.write(f"Number of anomalies detected: {outliers.shape[0]}")
            st.write("#### Anomalies Detected:")
            st.dataframe(outliers)

            # Clustering
            st.subheader("🌀 Clustering using KMeans")
            num_clusters = st.slider(
                "Select number of clusters",
                min_value=2, max_value=10, value=3, step=1
            )
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            df['Cluster'] = cluster_labels

            # Visualize Clusters
            st.write("### Cluster Visualization")
            pca = PCA(n_components=2)
            components = pca.fit_transform(data)
            df['Component 1'] = components[:, 0]
            df['Component 2'] = components[:, 1]
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                df['Component 1'], df['Component 2'],
                c=cluster_labels, cmap='viridis', alpha=0.7
            )
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title('KMeans Clustering Visualization')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            st.pyplot(fig)

            # NLP Processing
            st.subheader("💬 NLP Processing for Text Data")
            text_columns = df.select_dtypes(include=[object]).columns.tolist()
            if text_columns:
                text_column = st.selectbox("Select a text column for NLP processing", text_columns)
                if text_column:
                    st.write(f"#### Sample Text from '{text_column}':")
                    sample_text = df[text_column].dropna().astype(str).iloc[0]
                    st.write(sample_text)

                    # Prepare a prompt for OpenAI
                    prompt = f"""
                    You are a research assistant specialized in summarizing scientific texts.
                    Summarize the following text in a concise and informative way:
                    {sample_text}
                    """

                    # Generate summary using OpenAI API with a loading spinner
                    try:
                        with st.spinner('Generating summary...'):
                            chat_completion = client.chat.completions.create(
                                model="o1-mini",  # Use the appropriate model name
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=500
                            )
                            summary = chat_completion.choices[0].message.content
                        st.write(f"**Summary:** {summary}")
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.write("No text columns found for NLP processing.")

            # Option to download the processed dataset
            st.subheader("💾 Download Processed Dataset")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='processed_dataset.csv',
                mime='text/csv',
            )
        else:
            st.warning("Please select at least one numerical column for analysis.")
    else:
        st.info("Please upload a CSV file to proceed.")

with tab2:
    st.header("2. Hypothesis Generation and Testing Tool")
    st.subheader("Description")
    st.write("""
    An AI-driven tool that suggests potential hypotheses based on input datasets and previous research papers, and then helps design experiments or simulations to test those hypotheses.
    """)

    st.subheader("📂 Upload Your Dataset or Enter a Research Topic")
    uploaded_file_tab2 = st.file_uploader("Choose a CSV file", type="csv", key="tab2_file_uploader")
    user_input_tab2 = st.text_area("Or enter a research topic:", placeholder="e.g., Climate change impact on agriculture", key="tab2_text_area")

    if uploaded_file_tab2 is not None:
        # Read the dataset
        try:
            df = pd.read_csv(uploaded_file_tab2)
            st.write("### Dataset Preview")
            st.dataframe(df.head())

            # Provide an option for the user to select columns for hypothesis generation
            columns = df.columns.tolist()
            selected_columns = st.multiselect("Select column(s) for hypothesis generation:", columns, key="tab2_multiselect")

            if selected_columns:
                # Generate hypotheses using the AI model based on the selected columns data
                st.subheader("🔍 Generate Hypotheses from Data")
                try:
                    # Prepare a prompt for the AI based on the selected columns data
                    sample_data = df[selected_columns].dropna().head(5).to_dict(orient='records')
                    data_snippet = "\n".join([str(record) for record in sample_data])

                    prompt = f"""
                    You are an AI research assistant. Based on the following data samples, suggest 3-5 potential research hypotheses and experiments to test them:
                    Data Samples:
                    {data_snippet}
                    """

                    # Use a spinner while generating hypotheses
                    with st.spinner('Generating hypotheses...'):
                        chat_completion = client.chat.completions.create(
                            model="o1-mini",  # Use the appropriate model name
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500
                        )
                        hypotheses = chat_completion.choices[0].message.content
                    st.markdown(f"**Generated Hypotheses and Experiments:**\n\n{hypotheses}")
                except Exception as e:
                    st.error(f"Error generating hypotheses: {e}")
            else:
                st.info("Please select at least one column for hypothesis generation.")

        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

    elif user_input_tab2:
        # Generate hypotheses using the AI model based on user input topic
        st.subheader("🔍 Generate Hypotheses from Research Topic")
        try:
            # Prepare a prompt for the AI based on the user input topic
            prompt = f"""
            You are an AI research assistant with expertise in scientific research. Based on the following research topic, suggest 3-5 potential research hypotheses and experiments or simulations to test those hypotheses:
            Topic: {user_input_tab2}
            """

            # Use a spinner while generating hypotheses
            with st.spinner('Generating hypotheses...'):
                chat_completion = client.chat.completions.create(
                    model="o1-mini",  # Use the appropriate model name
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                hypotheses = chat_completion.choices[0].message.content
            st.markdown(f"**Generated Hypotheses and Experiments:**\n\n{hypotheses}")
            st.markdown("Under development for now, need to improve")
        except Exception as e:
            st.error(f"Error generating hypotheses: {e}")
    else:
        st.info("Please upload a dataset or enter a research topic to generate hypotheses and experiments.")

with tab3:
    st.header("3. AI-Powered Research Dashboard")
    st.subheader("Description")
    st.write("""
    A comprehensive dashboard integrating various AI tools for data analysis, model building, and reporting into a single platform. It serves as a one-stop solution for managing the entire research process.
    """)

    st.subheader("🎯 AI Research Tools Dashboard")
    st.write("Explore various AI tools for research directly through this dashboard:")

    # Use HTML with iframe for embedding external tools.
    st.markdown(
        """
        <style>
        .card {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
            margin-bottom: 20px;
        }
        .card:hover {
            transform: scale(1.02);
        }
        iframe {
            border-radius: 8px;
            border: none;
            width: 100%;
            height: 300px;
        }
        </style>
        
        <div class="card">
            <h4>Georgetown University - AI Research Guides</h4>
            <iframe src="https://guides.library.georgetown.edu/ai/tools" title="Georgetown AI Tools"></iframe>
        </div>
        
        <div class="card">
            <h4>Consensus - AI for Evidence-Based Research</h4>
            <iframe src="https://consensus.app/" title="Consensus AI"></iframe>
        </div>
        
        
        <div class="card">
            <h4>Elicit - AI Research Assistant</h4>
            <iframe src="https://elicit.org/" title="Elicit AI"></iframe>
        </div>

        <div class="card">
            <h4>ResearchRabbit - AI for Discovering Research</h4>
            <iframe src="https://researchrabbitapp.com/" title="ResearchRabbit"></iframe>
        </div>
        
        """, unsafe_allow_html=True
    )

    st.write("""
    Each card above represents an AI tool that can aid in different aspects of the research process, 
    from discovering and analyzing academic literature to generating evidence-based insights. Hover over the cards for a slight zoom effect!
    
             
    Explore other Research tools like: https://www.semanticscholar.org/ OR https://scite.ai/
             """)