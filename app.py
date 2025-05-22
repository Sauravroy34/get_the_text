import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ocr_label_converter import OCRLabelConverter

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, leakyRelu=False, alphabet=None):
        super(CRNN, self).__init__()
        ks = [3, 3, 3, 3]
        ps = [1, 1, 1, 1]
        ss = [1, 1, 1, 1]
        nm = [64, 128, 256, 512]
        self.cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            self.cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                self.cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leakyRelu:
                self.cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                self.cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        self.cnn.add_module('pooling0', nn.MaxPool2d((2, 2), stride=(2, 2)))
        convRelu(1)
        self.cnn.add_module('pooling1', nn.MaxPool2d((2, 2), stride=(2, 1)))
        convRelu(2, True)
        self.cnn.add_module('pooling2', nn.MaxPool2d((2, 2), stride=(2, 1)))
        convRelu(3, True)
        self.cnn.add_module('pooling3', nn.MaxPool2d((4, 1), stride=(4, 1)))
        self.cnn.add_module('conv_final', nn.Conv2d(512, 512, (1, 5), stride=(1, 1), padding=(0, 2)))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 81)
        )

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, f"Height must be 1, got {h}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        output = output.transpose(1, 0)
        return output




alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"""


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)  


st.title("Text Extraction from Images")
st.write("Upload an image to extract text using the CRNN model.")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    input_tensor = preprocess_image(image).to(device)

  

    model = CRNN().to(device)
    model = torch.load("ocr_model_final.pth", weights_only=False)
    model.eval()
    st.write("Model loaded successfully.")


    converter = OCRLabelConverter(alphabet)
    with torch.no_grad():
        logits = model(input_tensor).transpose(1, 0)
        logits = torch.nn.functional.log_softmax(logits, 2)
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for _ in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)

    st.write("**Extracted Text:**")
    st.write(sim_preds)


else:
    st.write("Please upload an image to proceed.")