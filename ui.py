import streamlit as st
import requests
from PIL import Image
import os

st.set_page_config(page_title="Sketch Retrieval Engine", layout="wide")

st.title("Advanced Sketch Retrieval Engine")
st.write("Powered by your local NVIDIA RTX A5000 GPU container.")

# Sidebar configurations
st.sidebar.header("Settings")
backend_url = st.sidebar.text_input("Backend API URL", "http://backend:8000/predict")

# File uploader
uploaded_file = st.file_uploader("Upload a Sketch Image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded sketch
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Sketch", width=300)
    
    if st.button("Search Similar Products"):
        with st.spinner("Processing sketch through PyTorch neural network..."):
            try:
                # Package the file to send to our FastAPI backend container
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(backend_url, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if the backend rejected the image (Outlier detection)
                    if data.get("status") == "error":
                        st.error(data.get("message"))
                    else:
                        st.success("Retrieval Complete!")
                        
                        matches = data.get("matches", [])  
                        
                        st.subheader("Top Matching Results:")
                        cols = st.columns(len(matches) if len(matches) > 0 else 1)
                        
                        for idx, img_name in enumerate(matches):
                            with cols[idx % len(cols)]:
                                st.write(f"Match #{idx+1}: {img_name}")
                                
                                full_image_path = f"/app/trainB/{img_name}"
                                
                                if os.path.exists(full_image_path):
                                    st.image(full_image_path, use_container_width=True)
                                else:
                                    st.warning("Image file not found.")
                else:
                    st.error(f"Backend returned an error code: {response.status_code}")
            
            
            except Exception as e:
                st.error(f"Could not connect to backend server: {e}")
