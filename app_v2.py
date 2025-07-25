import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import plotly.graph_objects as go

# Load the Sentence Transformer model only once for efficiency
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Function to process files and generate the final output DataFrame
def process_files(client_file, taxonomy_file):
    model = load_model()

    # Load data from uploaded files
    client_categories = pd.read_excel(client_file)
    std_taxonomy = pd.read_excel(taxonomy_file)

    # Preprocess the data
    client_categories[['Level_1', 'Level_2', 'Level_3', 'Level_4']] = client_categories['Client categories'].str.split('>', expand=True)
    std_taxonomy[['Standard_Level_1', 'Standard_Level_2', 'Standard_Level_3', 'Standard_Level_4']] = std_taxonomy['Std_taxonomy'].str.split('>', expand=True)

    # Compute embeddings
    client_embeddings = model.encode(client_categories['Client categories'].tolist())
    std_embeddings = model.encode(std_taxonomy['Std_taxonomy'].tolist())

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(client_embeddings, std_embeddings)

    # Map each row in client_categories to the closest match in std_taxonomy
    client_categories['Closest_Match_Index'] = similarity_matrix.argmax(axis=1)

    # Collect data for the final output
    final_output_data = []
    for idx, row in client_categories.iterrows():
        closest_match_idx = row['Closest_Match_Index']
        closest_match = std_taxonomy.iloc[closest_match_idx]

        final_output_data.append({
            'Level_1': row['Level_1'],
            'Level_2': row['Level_2'],
            'Level_3': row['Level_3'],
            'Level_4': row['Level_4'],
            'Standard_Level_1': closest_match['Standard_Level_1'],
            'Standard_Level_2': closest_match['Standard_Level_2'],
            'Standard_Level_3': closest_match['Standard_Level_3'],
            'Standard_Level_4': closest_match['Standard_Level_4'],
        })

    final_output = pd.DataFrame(final_output_data)
    return final_output

# Function to compare manual mapping with generated output
def compare_mappings(generated_df, manual_df):
    # Merge data on hierarchical levels to check for exact matches

    comparison_df = pd.merge(
        generated_df,
        manual_df,
        on=['Standard Level_1', 'Standard Level_2', 'Standard Level_3', 'Standard Level_4', 'Level_1', 'Level_2', 'Level_3', 'Level_4', ],
        how='outer',
        indicator=True
    )

    # Calculate similarity score as the proportion of matched rows
    total_rows = len(comparison_df)
    matched_rows = len(comparison_df[comparison_df['_merge'] == 'both'])
    similarity_score = (matched_rows / total_rows) * 100

    # Create a DataFrame of mismatches for review
    mismatches = comparison_df[comparison_df['_merge'] != 'both']
    mismatches = mismatches.drop(columns=['_merge'])

    return similarity_score, mismatches

# Function to create a circular progress indicator for similarity score
def show_similarity_gauge(similarity_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=similarity_score,
        title={'text': "Similarity Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
        }
    ))
    st.plotly_chart(fig)

# Streamlit UI
st.title("Category Mapping Application")
st.write("Upload the required files below to begin mapping client categories to the standard taxonomy.")

# File upload interface
client_file = st.file_uploader("Upload Client Categories Excel File", type="xlsx")
taxonomy_file = st.file_uploader("Upload Standard Taxonomy Excel File", type="xlsx")
manual_mapping_file = st.file_uploader("Upload Manual Mapping File (for comparison)", type="xlsx")

# Check if both files are uploaded before processing
if client_file and taxonomy_file:
    st.write("Files uploaded successfully! Processing...")

    # Process the files and generate the output DataFrame
    with st.spinner("Calculating category mappings..."):
        final_output = process_files(client_file, taxonomy_file)
        st.success("Processing complete!")

        # Display a preview of the final output
        st.write("Preview of the Final Output:")
        st.dataframe(final_output.head())

        # Generate downloadable Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            final_output.to_excel(writer, index=False, sheet_name='Final_Output')

        # Set the download button
        st.download_button(
            label="Download Output as Excel",
            data=output.getvalue(),
            file_name="Final_Output_v3.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # If manual mapping file is uploaded, compare it with the generated output
        if manual_mapping_file:
            st.write("Manual mapping file uploaded. Comparing with generated mappings...")

            # Load manual mapping file
            manual_mapping = pd.read_excel(manual_mapping_file)

            # Calculate similarity score and mismatches
            similarity_score, mismatches = compare_mappings(final_output, manual_mapping)

            # Show circular similarity gauge
            show_similarity_gauge(similarity_score)

            # Show mismatches, if any
            if not mismatches.empty:
                st.write("Mismatches found between generated output and manual mapping:")
                st.dataframe(mismatches)
            else:
                st.write("No mismatches found. The generated output matches the manual mapping completely.")
else:
    st.write("Please upload both the Client Categories and Standard Taxonomy files to proceed.")
