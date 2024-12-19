import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from DCGAN_ResNet import DCGANTrainer 
from hyperbolic_mscnn import HyperbolicMSCNN, manifold 
from hypll.tensors import TangentTensor
from SegmentationCrop import process_and_crop_segmented_image
from deeplabv3_plus_ import DeepLabV3Plus
import time
from io import BytesIO
from fpdf import FPDF
import os

def save_as_pdf(image_name, prediction_result, model_choice, image_bytes, output_pdf="report.pdf"):
    """
    Generates a PDF report containing the image (centered and resized), prediction result, and model information.

    Args:
        image_name (str): Name of the uploaded or captured image.
        prediction_result (str): The predicted defect classification ("Defect" or "No Defect").
        model_choice (str): The selected defect detection model ("DCGAN ResNet" or "Hyperbolic MSCNN").
        image_bytes (bytes): Bytes representing the image data.
        output_pdf (str): Path to save the generated PDF file.
    """
    try:
        # Initialize FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Arial", size=16, style="B")
        pdf.cell(0, 10, "Defect Detection Report", ln=True, align="C")

        pdf.ln(10)  # Add some vertical space

        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as img_file:
            img_file.write(image_bytes)

        # Open image to get its dimensions
        with Image.open(temp_image_path) as image:
            image_width, image_height = image.size

            # Resize the image to fit within a maximum width/height while maintaining the aspect ratio
            max_width_mm = 150  # Maximum width for the image in mm
            max_height_mm = 100  # Maximum height for the image in mm
            aspect_ratio = image_width / image_height

            if image_width > image_height:
                display_width_mm = min(max_width_mm, pdf.w - 20)  # Leave margins
                display_height_mm = display_width_mm / aspect_ratio
            else:
                display_height_mm = min(max_height_mm, pdf.h / 2)  # Adjust height to fit within half page
                display_width_mm = display_height_mm * aspect_ratio

            # Center the image on the page
            x_centered = (pdf.w - display_width_mm) / 2

            # Add the image to the PDF
            pdf.image(temp_image_path, x=x_centered, y=pdf.get_y(), w=display_width_mm, h=display_height_mm)
            pdf.ln(display_height_mm + 10)  # Adjust line height to move to next section

        # Table Header
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(60, 10, "Field", border=1, align="C")
        pdf.cell(120, 10, "Value", border=1, ln=True, align="C")

        # Table Rows
        pdf.set_font("Arial", size=12)
        table_data = [
            ["Image Name", image_name],
            ["Prediction Result", prediction_result],
            ["Defect Detection Model", model_choice],
        ]

        for row in table_data:
            pdf.cell(60, 10, row[0], border=1, align="L")
            pdf.cell(120, 10, row[1], border=1, ln=True, align="L")

        # Save the PDF
        pdf.output(output_pdf)

        # Cleanup: Remove the temporary image
        os.remove(temp_image_path)

        return output_pdf

    except Exception as e:
        raise RuntimeError(f"Error generating PDF: {e}")

# Paths to model checkpoints
seg_model_path = 'C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/fyp/model/segmentation_model.pth'
dcgan_defect_model_path = 'C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/fyp/model/dcganResnet_discriminator_6b.pth'
hyperbolic_defect_model_path = 'C:/Users/User/OneDrive/Documents/Wilson/TARUMT/Degree/Year3/fyp/model/hyperbolic_mscnn_gc(lr2)_.pth'

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the segmentation model
segmentation_model = DeepLabV3Plus(num_classes=9)
segmentation_model.load_state_dict(torch.load(seg_model_path, map_location=device))
segmentation_model.to(device)
segmentation_model.eval()

# Load DCGAN Discriminator
dcgan_discriminator = DCGANTrainer.load_model(DCGANTrainer.Discriminator, model_path=dcgan_defect_model_path)

# Load Hyperbolic MSCNN
hyperbolic_mscnn = HyperbolicMSCNN(input_channels=3, num_classes=2)
hyperbolic_mscnn.load_state_dict(torch.load(hyperbolic_defect_model_path, map_location=device))
hyperbolic_mscnn.to(device)
hyperbolic_mscnn.eval()

# Define image transformations
transform_inference = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction function for DCGAN
def predict_defect_dcgan(image, model):
    image_tensor = transform_inference(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[1]  # Extract logits if the output is a tuple
        prediction = torch.sigmoid(output).item()  # Binary classification
    return prediction

# Prediction function for Hyperbolic MSCNN
def predict_defect_hyperbolic(image, model):
    image_tensor = transform_inference(image).unsqueeze(0).to(device)
    tangents = TangentTensor(data=image_tensor, man_dim=1, manifold=model.manifold)
    manifold_image = manifold.expmap(tangents)
    with torch.no_grad():
        output = model(manifold_image)
        output_tensor = output.tensor
        if output_tensor.shape[1] == 2:
            prediction = torch.sigmoid(output_tensor[:, 1]).item()
        else:
            prediction = torch.sigmoid(output_tensor).max(1).values.item()
    return prediction

# Streamlit UI
st.image("C:/Users/User/Downloads/Banner.png")

# Layout using columns
col1, col2 = st.columns(2)

# Model Selection and Image Input Method
with col1:
    model_choice = st.radio("Select defect detection model:", ("DCGAN ResNet", "Hyperbolic MSCNN"))

with col2:
    input_choice = st.radio("Select the image input method:", ("Upload Image", "Use Camera"))

# Track the last uploaded or captured image to reset session state when a new image is uploaded
if "last_image" not in st.session_state:
    st.session_state.last_image = None  # Stores the last uploaded/captured image as bytes
# Check if session state keys exist; initialize if not
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store processing messages

# Handle input
input_image = None
img_name = None
image_changed = False  # Flag to track if the input image has changed
if input_choice == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_image:
        input_image = Image.open(uploaded_image).convert("RGB")
        # Convert the image to bytes
        image_bytes = input_image.tobytes()
        img_name = "uploaded_image.jpg"

         # Check if the uploaded image is different from the previous one
        current_image_bytes = uploaded_image.getvalue()
        if st.session_state.last_image != current_image_bytes:
            st.session_state.last_image = current_image_bytes
            image_changed = True  # Mark that the image has changed
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        input_image = Image.open(BytesIO(camera_image.getvalue())).convert("RGB")  # Handle BytesIO for camera input
        img_name = "captured_image.jpg"

        # Check if the captured image is different from the previous one
        current_image_bytes = camera_image.getvalue()
        if st.session_state.last_image != current_image_bytes:
            st.session_state.last_image = current_image_bytes
            image_changed = True  # Mark that the image has changed

# Reset session state if the image has changed
if image_changed:
    st.session_state.prediction_result = None
    st.session_state.report_ready = False
    st.session_state.messages = []

# Main layout for image processing and PDF generation
if input_image:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.image(input_image, caption="Your Input Image")

    with col_right:
        try:
            # Step 1: Process the image with the segmentation model
            if st.session_state.prediction_result is None:
                st.session_state.messages = []
                st.session_state.messages.append("DeepLabV3+ processing segmentation...")
                st.write("DeepLabV3+ processing segmentation...")

                cropped_image = process_and_crop_segmented_image(
                    model=segmentation_model,
                    image=input_image,
                    device=device,
                    target_size=(256, 256),
                    mask_color=(255, 255, 255),
                    exclude_classes=[2, 4, 6, 7]
                )
                st.session_state.messages.append("DeepLabV3+ done segmentation.")
                st.write("DeepLabV3+ done segmentation.")

                # Step 2: Predict defect based on selected model
                st.session_state.messages.append(f"{model_choice} is predicting defect...")
                st.write(f"{model_choice} is predicting defect...")
                bar = st.progress(0)

                if model_choice == "DCGAN ResNet":
                    prediction = predict_defect_dcgan(cropped_image, dcgan_discriminator)
                else:
                    prediction = predict_defect_hyperbolic(cropped_image, hyperbolic_mscnn)

                time.sleep(3)
                bar.progress(100)
                # Save prediction result
                st.session_state.prediction_result = "Defect" if prediction > 0.6 else "No Defect"
            else:
                for msg in st.session_state.messages:
                    st.write(msg)

            result = st.session_state.prediction_result
            st.success("Prediction done successfully.")
            st.info(f"Your Clothes Prediction Result is: {result}")
            if result == "Defect":
                st.info("Refund request is approved, will be processing in 1-3 working days.")
            else:
                st.info("Refund request is denied as there is no defect being found from the clothes.")

            # Buttons to save PDF and download
            col_buttons = st.columns([1, 1])

            with col_buttons[0]:
                if st.button("Save Report as PDF"):
                    buffer = BytesIO()
                    input_image.save(buffer, format="JPEG")
                    image_bytes = buffer.getvalue()

                    # Generate PDF and get the path
                    pdf_path = save_as_pdf(
                        image_name=img_name,
                        prediction_result=st.session_state.prediction_result,
                        model_choice=model_choice,
                        image_bytes=image_bytes
                    )

                    # Store pdf_path in session state
                    if pdf_path:
                        st.session_state.pdf_path = pdf_path
                        st.session_state.report_ready = True
                        
            with col_buttons[1]:
                if st.session_state.report_ready:
                    # Ensure pdf_path is set before allowing download
                    if 'pdf_path' in st.session_state and os.path.exists(st.session_state.pdf_path):
                        with open(st.session_state.pdf_path, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                        st.download_button(
                            label="Download PDF Report",
                            data=PDFbyte,
                            file_name="defect_report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("PDF report not found. Please generate the PDF first.")

        except Exception as e:
            st.error(f"Error processing the image: {e}")