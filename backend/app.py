import os
from os.path import abspath, dirname, join
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, make_response, session
from reverseProxy import proxyRequest
from werkzeug.utils import secure_filename
from flask_session import Session
import json

import torch
from torch import nn
import cv2
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import sys
import torch.nn.functional as F
import pickle
import shutil


MODE = os.getenv('FLASK_ENV')
DEV_SERVER_URL = 'http://localhost:3000/'
basedir = abspath(dirname(__file__))

app = Flask(__name__)
app.secret_key = 'badabingbadaboom'
# app.config['SECRET_KEY'] = 'dakey'

if MODE == "development":
    app = Flask(__name__, static_folder=None)
    app.secret_key = 'badabingbadaboom'

api_routes = ['upload', 'next_image', 'prev_image', 'download', 'clear', 'get_image', 'get_mask']
upload_img_path = "C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/uploads/"
processed_img_path = "C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/processed/"
download_path = "C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/zipped/"


@app.route('/')
@app.route('/<path:path>')
def index(path=''):
    if path in api_routes:
        print('path accessed: ', path)
        return redirect(url_for(path))
    session['user'] = 'user'
    session['curr_idx'] = 0
    session['filenames'] = []
    return proxyRequest(DEV_SERVER_URL, path)


@app.route('/upload', methods = ['POST'])
def upload():
    files = request.files.getlist("file")
    imagew = int(request.form['imagew'])
    imageh = int(request.form['imageh'])
    filenames = session['filenames']

    for f in files:
        filenames.append(f.filename)
        f.save(os.path.join(upload_img_path, secure_filename(f.filename)))

    session['filenames'] = filenames
    
    batch_inference(imagew=imagew, imageh=imageh) # infer all the images in the upload folder

    if len(session['filenames']) == 0:
        return make_response("upload: error")
    return make_response("upload: success")
    


@app.route('/get_image/<idx>', methods = ['GET'])
def get_image(idx):
    print("Image: ", session['curr_idx'], flush=True)
    print(session['filenames'], flush=True)
    response = make_response(send_file(join(upload_img_path, session['filenames'][int(idx)])))
    response.headers['Content-Transfer-Encoding']='base64'
    return response

@app.route('/get_mask/<idx>', methods = ['GET'])
def get_mask(idx):
    print("mask: ", session['curr_idx'], flush=True)
    response = make_response(send_file(join(processed_img_path, session['filenames'][int(idx)])))
    response.headers['Content-Transfer-Encoding']='base64'
    return response


@app.route('/next_image', methods = ['GET'])
def next_image():
    if session['curr_idx'] == len(session['filenames']) -1:
        return "error"
    ci = session['curr_idx']
    ci += 1
    session['curr_idx'] = ci
    print("after next: ", session['curr_idx'], flush=True)
    return make_response("success")


@app.route('/prev_image', methods = ['GET'])
def prev_image():
    if session['curr_idx'] - 1 < 0:
        return "error"
    ci = session['curr_idx']
    ci -= 1
    session['curr_idx'] = ci
    print("after prev: ", session['curr_idx'], flush=True)
    return make_response("success")


@app.route('/download', methods = ['GET'])
def download():
    return send_file(join(download_path, "results.zip"))


@app.route('/clear', methods = ['GET'])
def clear():
    clear_files()
    return "All Clear" 


def clear_files():
    session['filenames'] = []
    session['curr_idx'] = 0
    for f in os.listdir("C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/uploads/"):
        os.remove(os.path.join("C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache", f))
    for f in os.listdir("C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/processed/"):
        os.remove(os.path.join("C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache", f))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = torch.sigmoid(output)
            return output


def batch_inference(imagew, imageh):
    modelpath ="./deserialized.pth"
    uploads_path ="C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/uploads/"
    processed_path ="C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/processed/"

    a_file = open("C:/Users/jnynt/Desktop/AifSR/webapp/backend/themodel.pkl", "rb")
    model_state = pickle.load(a_file)

    net = NestedUNet()
    net.load_state_dict(model_state)
    net.eval()
    data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    
    for (root, dirs, files) in os.walk(uploads_path, topdown=True):
        for f in files:
            img = cv2.imread(os.path.join(uploads_path, f))
            mask = infer_once(net, img, data_transforms)
            mask = cv2.resize(mask, (imagew, imageh), interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(processed_path, f), mask)

    shutil.make_archive("C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/zipped/results", 'zip', "C:/Users/jnynt/Desktop/AifSR/webapp/backend/image_cache/processed")


def infer_once(net, image, data_transforms):
    torch.no_grad()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(gray)

    imgblob = data_transforms(image).unsqueeze(0)
    predict = net(imgblob).cpu().data.numpy().copy()
    predict = predict > 0.5
    result = np.squeeze(predict)

    result = (result*255).astype(np.uint8)
    resultimage = image.copy()
    return result





if __name__ == "__main__":
    app.run()



