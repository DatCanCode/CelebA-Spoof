import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import insightface
from tsn_predict import TSNPredictor as CelebASpoofDetector
import matplotlib.pyplot as plt

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title("Demo face anti-spoofing detection")

    run_the_app()


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    st.sidebar.header("Load image")
    uploaded_file = st.sidebar.file_uploader("Select a photo", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.header("Input image")
        st.image(image, caption="Sometimes it was rotated but I don't know why", use_column_width=True)
        
        img_array = load_image(image)
        
        st.header("Cropped face")
        img_face = crop_face(img_array)
        if not img_face[0]:
            if img_face[1] == "No face":
                st.write("There is no face in the image.")
            elif img_face[1] == "Too much faces":
                st.write("There are too much faces!!!")
            else:
                st.write("Cannot crop the face")
        else:
            st.image(img_face[1], caption='Cropped face was rotated... occasionally ¯\_(ツ)_/¯', use_column_width=True)
            prob = detect_spoof(img_face[1])

            st.header("And I think that...")
            draw_result(prob)

            
def draw_result(prob):
    data = pd.DataFrame(prob[0], columns=["AENET"], index=['Live', "Spoof"])

    st.write(data)
    fig, ax = plt.subplots()
    ax.barh(data.index, width=data.AENET, color=['g', 'r'], height=.5)
    st.pyplot(fig)

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(img):
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def crop_face(image):
    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network():
        model = insightface.model_zoo.get_model('retinaface_r50_v1')
        model.prepare(ctx_id = -1, nms=0.4)
        return model
    model = load_network()

    scale = 1
    px = image.shape[0] + image.shape[1]
    if px > 1400:
        scale = 1400/px
    bbox, landmark = model.detect(image, threshold=0.5, scale=scale)

    if len(bbox) == 0:
        return False, "No face"
    elif len(bbox) > 1:
        return False, "Too much faces"
    else:
        x1, y1, x2, y2, score = [str(int(p)) if (p > 1) else str(p) for p in bbox[0]]
        xx1 = 0 if x1[0] == '-' else int(x1)
        yy1 = 0 if y1[0] == '-' else int(y1)
        yy2 = int(y2)
        xx2 = int(x2)

    try:
        # st.write(yy1,yy2,xx1,xx2)
        img = image[yy1:yy2,xx1:xx2,:]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return False, "Cannot crop face"

    return True, img


def detect_spoof(face_img):
    @st.cache(allow_output_mutation=True)
    def load_network():
        model = CelebASpoofDetector()
        return model
    
    detector = load_network()

    prob = detector.predict(face_img[None])
    return prob


if __name__ == "__main__":
    main()
