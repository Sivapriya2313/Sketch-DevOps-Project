# 🎨 Advanced AI Sketch Retrieval Engine

A full-stack, containerized Deep Learning DevOps pipeline for real-time Sketch-Based Image Retrieval (SBIR). 

This project allows a user to upload a hand-drawn sketch of a chair, processes it through a custom-trained PyTorch neural network, and instantly retrieves the top 10 physically matching real-world product photos from a dataset.

### 🚀 Live Architecture Overview
This pipeline is built using a modern microservice architecture, separating the heavy GPU computations from the user interface, and wrapping everything in Docker for seamless deployment.

* **Backend Engine (FastAPI):** A high-performance REST API that manages the active GPU state, handles image preprocessing, and executes tensor mathematics.
* **Frontend Dashboard (Streamlit):** A reactive web interface for user sketch uploads and dynamic visual product rendering.
* **DevOps (Docker Compose):** Fully containerized multi-container architecture with internal network bridging and seamless host-to-container volume mounting.

### 🧠 Core AI Features

#### 1. On-the-Fly Runtime Indexing
Instead of relying on static, pre-calculated embeddings, the backend dynamically builds its own database upon boot. It ingests hundreds of physical dataset images, passes them through the neural network to extract high-dimensional mathematical vectors, and indexes them directly into RAM using Cosine Similarity calculations.

#### 2. Out-of-Distribution (OOD) Detection
To prevent the mathematical anomalies that occur when feeding untrained objects (like shoes or cars) into a domain-specific model, this engine features a strict mathematical validation layer. By calculating the Cosine Similarity distance of the input vector against the dataset, the API relies on a confidence threshold to safely reject non-domain inputs before returning invalid product matches.

#### 3. Custom Phase-2 Retrieval Model
The core matching engine utilizes a custom-trained `Phase2ResNet50` architecture, fine-tuned specifically to bridge the visual domain gap between abstract human line drawings and photorealistic product images.

---

### 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **API Framework:** FastAPI, Uvicorn
* **Frontend:** Streamlit, Requests
* **Hardware Acceleration:** NVIDIA CUDA (RTX A5000)
* **Deployment:** Docker, Docker Compose

---

### ⚙️ How to Run Locally

**Prerequisites:** You must have Docker and Docker Desktop installed, along with an NVIDIA GPU and the NVIDIA Container Toolkit. QUML Chair_V2 dataset. 

**1. Clone the repository**
```bash
git clone [https://github.com/Sivapriya2313/Sketch-DevOps-Project.git](https://github.com/Sivapriya2313/Sketch-DevOps-Project.git)
cd Sketch-DevOps-Project
