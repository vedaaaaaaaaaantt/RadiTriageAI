import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from fpdf import FPDF
from datetime import datetime
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- DARK THEME ----------------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* Text */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #ffffff !important;
}

/* Buttons */
.stButton>button {
    background-color: #1f77ff;
    color: white;
    border-radius: 10px;
    border: none;
}

/* File uploader */
.stFileUploader {
    background-color: #1c1f26;
    border-radius: 10px;
    padding: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161a23;
}

/* Cards / containers */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.markdown("""
    <div style="
        background-color:#1c1f26;
        padding:10px;
        border-radius:12px;
        display:inline-block;
    ">
    """, unsafe_allow_html=True)

    st.image("logo.png", width=80)

    st.markdown("</div>", unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0;'>RadiTriageAI</h1>
    <p style='color:gray;'>⚡ Instant AI triage for chest X-rays | Built for Doctors</p>
    """, unsafe_allow_html=True)

st.warning("⚠️ AI decision support only. Not a substitute for medical diagnosis.")

# ---------------- UTIL ----------------
def clean_text(text):
    return text.encode("latin-1", "ignore").decode("latin-1")

def generate_pdf(name, patient_id, predictions, risk, urgency):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="RadiTriageAI Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=clean_text(f"Patient Name: {name}"), ln=True)
    pdf.cell(200, 10, txt=clean_text(f"Patient ID: {patient_id}"), ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt=clean_text(f"Risk Score: {round(risk, 3)}"), ln=True)
    pdf.cell(200, 10, txt=clean_text(f"Urgency: {urgency}"), ln=True)
    pdf.ln(10)

    pdf.cell(200, 10, txt="Top Predictions:", ln=True)

    for label, conf in predictions:
        pdf.cell(200, 10, txt=clean_text(f"{label}: {round(conf, 3)}"), ln=True)

    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path

# ---------------- STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = ""

DEVICE = torch.device("cpu")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩻 RadiTriageAI")

if st.session_state.logged_in:
    st.sidebar.caption(f"Logged in as: Dr. {st.session_state.doctor_name}")
else:
    st.sidebar.caption("Doctor Panel")

page = st.sidebar.radio("Navigation", ["Dashboard", "History"])

if st.session_state.logged_in:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.doctor_name = ""
        st.rerun()



CLASS_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

def load_model():
    MODEL_URL = "https://drive.google.com/uc?id=1d2WvyvgUYZe8AKDtHrfUCNrJfOXCnZ8c"
    MODEL_PATH = "model.pth"

    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

    # ✅ USE RESNET (NOT DENSENET)
    model = models.resnet50(pretrained=False)

    # change final layer
    model.fc = nn.Linear(model.fc.in_features, 14)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ---------------- LOAD MODEL ----------------
model = load_model()

# ---------------- GRAD CAM ----------------
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# ✅ NOW attach hooks (after model exists)
model.layer4.register_forward_hook(forward_hook)
model.layer4.register_backward_hook(backward_hook)

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- PROCESS ----------------
def process_image(image):
    global features, gradients

    img = transform(image).unsqueeze(0).to(DEVICE)

    output = model(img)
    probs = torch.sigmoid(output)[0]

    top3 = torch.topk(probs, 3)

    predictions = []
    for i in range(3):
        idx = top3.indices[i].item()
        conf = top3.values[i].item()
        predictions.append((CLASS_NAMES[idx], round(conf, 3)))

    model.zero_grad()
    target_class = top3.indices[0]
    output[0, target_class].backward(retain_graph=True)

    # ✅ safety check
    if gradients is None or features is None:
        st.error("Grad-CAM failed")
        return predictions, np.zeros((224,224,3)), 0, "ERROR"

    weights = gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * features).sum(dim=1).squeeze()

    cam = torch.relu(cam).detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    if cam.max() != 0:
        cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(image.resize((224, 224)))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    risk = top3.values[0].item()

    def get_urgency(score):
        if score > 0.85:
            return "🚨 CRITICAL"
        elif score > 0.6:
            return "⚠️ HIGH"
        elif score > 0.3:
            return "MODERATE"
        else:
            return "LOW"

    urgency = get_urgency(risk)

    return predictions, overlay, risk, urgency

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:

    st.title("👨‍⚕️ Doctor Login")
    st.caption("Secure access to RadiTriageAI")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    doctors = {
        "Vedant": "1234",
        "Rushikesh": "4444",
        "Shivraj": "0303"
    }

    if st.button("Login"):
        if username in doctors and doctors[username] == password:
            st.session_state.logged_in = True
            st.session_state.doctor_name = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.markdown("## 🩻 RadiTriageAI")
    st.caption("AI-powered Chest X-ray Triage")

    st.subheader("📊 Live Analytics")

    total_cases = len(st.session_state.history)
    critical = sum(1 for c in st.session_state.history if "CRITICAL" in c["urgency"])
    high = sum(1 for c in st.session_state.history if "HIGH" in c["urgency"])

    a1, a2, a3 = st.columns(3)

    a1.metric("📁 Total Cases", total_cases)
    a2.metric("🚨 Critical", critical)
    a3.metric("⚠️ High Risk", high)


    # -------- TOP SECTION --------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Patient Information")
        patient_name = st.text_input("Patient Name", placeholder="Enter name")
        patient_id = st.text_input("Patient ID", placeholder="Enter ID")

    with col2:
        st.markdown("### X-Ray Images")

        col_upload, col_camera = st.columns([2, 1])

        # -------- FILE UPLOAD --------
        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload X-Rays",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

        # -------- CAMERA (FIXED TOGGLE) --------
        with col_camera:

            if "camera_on" not in st.session_state:
                st.session_state.camera_on = False

            if not st.session_state.camera_on:
                if st.button("📷 Open Camera"):
                    st.session_state.camera_on = True

            camera_image = None

            if st.session_state.camera_on:
                camera_image = st.camera_input("Capture Image")

                if camera_image is not None:
                    st.success("Image Captured ✅")

                    if st.button("Close Camera"):
                        st.session_state.camera_on = False

    # -------- MERGE INPUTS --------
    all_images = []

    if uploaded_files:
        for file in uploaded_files:
            all_images.append({
                "name": file.name,
                "image": Image.open(file).convert("RGB")
            })

    if camera_image is not None:
        all_images.append({
            "name": "Camera Capture",
            "image": Image.open(camera_image).convert("RGB")
        })

    # -------- ANALYZE BUTTON --------
    analyze = st.button(f"⚡ Analyze All ({len(all_images)})")

    # -------- EMPTY STATE --------
    if not analyze:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:gray;'>
        Upload X-ray images and click <b>Analyze All</b> to begin
        </div>
        """, unsafe_allow_html=True)

    # -------- RESULTS --------
    if analyze and all_images:

        results = []

        for item in all_images:
            image = item["image"]

            with st.spinner(f"Analyzing {item['name']}..."):
                predictions, heatmap, risk, urgency = process_image(image)

            results.append({
                "file_name": item["name"],
                "image": image,
                "heatmap": heatmap,
                "risk": risk,
                "urgency": urgency,
                "predictions": predictions
            })

        # 🔥 SORT BY RISK
        results = sorted(results, key=lambda x: x["risk"], reverse=True)

        st.markdown("## 🚨 Priority Queue")

        for result in results:

            st.markdown("---")

            colA, colB, colC = st.columns([1, 1, 2])

            # -------- X-RAY --------
            with colA:
                st.markdown("**X-Ray Image**")
                st.image(result["image"], use_container_width=True)

            # -------- HEATMAP --------
            with colB:
                st.markdown("**Heatmap**")
                st.image(result["heatmap"], channels="BGR", use_container_width=True)

            # -------- RIGHT PANEL --------
            with colC:

                st.markdown("### Risk Assessment")

                st.markdown(f"""
                <h1 style='color:#ff4b4b;'>{int(result['risk']*100)}/100</h1>
                """, unsafe_allow_html=True)

                st.progress(float(result["risk"]))

                if result["risk"] > 0.85:
                    st.error("🚨 CRITICAL")
                elif result["risk"] > 0.6:
                    st.warning("⚠️ HIGH")
                elif result["risk"] > 0.3:
                    st.info("MODERATE")
                else:
                    st.success("LOW")

                st.markdown("### AI Predictions")

                for label, conf in result["predictions"]:
                    st.write(f"{label} ({int(conf*100)}%)")
                    st.progress(float(conf))

            st.session_state.history.append({
                "name": patient_name,
                "id": patient_id,
                "risk": round(result["risk"], 3),
                "urgency": result["urgency"],
                "top_prediction": result["predictions"][0][0],
                "image": result["image"],
                "heatmap": result["heatmap"],
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            })
            
            # -------- PDF --------
            pdf_path = generate_pdf(
                patient_name,
                patient_id,
                result["predictions"],
                result["risk"],
                result["urgency"]
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name=f"{result['file_name']}_report.pdf",
                    mime="application/pdf"
                )

# ---------------- HISTORY ----------------
elif page == "History":

    st.subheader("📋 Case History")
    
    # -------- EXPORT CSV --------
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Full History CSV",
            data=csv,
            file_name="radi_triage_history.csv",
            mime="text/csv"
        )

    search = st.text_input("🔍 Search Patient Name or ID")

    if st.session_state.history:

        for case in reversed(st.session_state.history):

            if search:
                if search.lower() not in case["name"].lower() and search.lower() not in case["id"].lower():
                    continue

            st.markdown(f"### 👤 {case['name']} ({case['id']})")
            st.write(f"🕒 {case['timestamp']}")
            st.write(f"🩻 {case['top_prediction']}")
            st.write(f"⚠️ {case['urgency']} | Risk: {case['risk']}")

            col1, col2 = st.columns(2)

            with col1:
                st.image(case["image"], caption="Original Image", width=220)

            with col2:
                st.image(case["heatmap"], caption="Heatmap", width=220, channels="BGR")
            st.markdown("---")

    else:
        st.write("No cases yet")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='font-size:12px; color:gray; text-align:center;'>
⚠️ This platform is intended for clinical decision support only. It does not replace professional medical judgment. 
Always consult a qualified healthcare provider before making diagnostic or treatment decisions.
</p>
""", unsafe_allow_html=True)

