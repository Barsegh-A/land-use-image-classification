import adddeps

import streamlit as st
import cv2
import io
import re
import numpy as np
import torch
import torchvision.transforms as T

from src.dataset import CLASSES
from src.models import get_multilabel_model
from src.utils import inference

def load_model(model_file, model_path):
    out = re.findall("[a-z]*net[0-9]*", model_path)
    if len(out) == 0:
        model_name = 'resnet18'
    else:
        model_name = out[0]
    num_classes = len(CLASSES)
    model = get_multilabel_model(model_name, num_classes=num_classes, weights=None)

    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    return model

class App(object):
    def __init__(self):
        self.conf_threshold = 0.5
        self.model = None
        self.classes = []
        self.input_height = 256
        self.input_width = 256

    def process_image(self, uploaded_file):
        transform = T.Compose([
            T.Resize((self.input_height, self.input_width)),
            T.ToTensor()
        ])
        labels = inference(uploaded_file, self.model, transform, self.conf_threshold)
        return labels

    def create_sidebar(self):
        st.sidebar.title('Configuration')
        self.conf_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5)

        st.sidebar.title('Model')
        uploaded_file = st.sidebar.file_uploader('Choose model file', type=['pt'])
        if uploaded_file:
            try:
                self.model = load_model(uploaded_file, uploaded_file.name)
                st.sidebar.success('Model is successfully loaded!')
                with st.sidebar.expander('Classes'):
                    st.text('\n'.join(CLASSES))
            except Exception as e:
                st.sidebar.error('Can not load model!')

    def process_uploaded_files(self, uploaded_files):
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if uploaded_file.type[:5] == 'image':
                labels = self.process_image(uploaded_file)
                labels_str = ', '.join(labels[0])
                st.subheader(f'Predicted labels: {labels_str}')
                st.image(uploaded_file, channels='BGR')
            else:
                st.error('Unsuppored file.')

    def create_main_container(self):
        st.title('Land use multilabel classification')
        with st.form('classification form'):
            uploaded_files = st.file_uploader('Choose files', accept_multiple_files=True, type=['jpg', 'png'])
            submitted = st.form_submit_button("Process")
            if submitted:
                if len(uploaded_files) > 0:
                    if not self.model:
                        st.error('Model is not loaded!')
                        return

                    placeholder = st.empty()
                    with st.spinner('Processing'):
                        self.process_uploaded_files(uploaded_files)
                    st.success('Done!')
                else:
                    st.error('No files are uploaded! Please upload files.')

    def run(self):
        self.create_sidebar()
        self.create_main_container()

if __name__ == '__main__':
    app = App()
    app.run()
