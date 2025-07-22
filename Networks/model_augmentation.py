from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.UNet_test import *

import numpy as np
np.random.seed(2885)
import os
import copy
import pickle
import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]


        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = UNet(param).to(self.device) #initialisation of the class inside basicNetwork.py

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = OxfordPetDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = OxfordPetDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = OxfordPetDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)


    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
            self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl',map_location=torch.device('cpu'))) # map_location can be cancelled with gpu I guess

    # -----------------------------------
    # TRAINING LOOP (fool implementation)
    # -----------------------------------

    def train(self): 
        # train for a given number of epochs
        losses = []
        mean_losses = []
        validation_losses = []
        mean_validation_loss = []
        for i in range(self.epoch):
            self.model.train()
            modelWts = copy.deepcopy(self.model.state_dict())
            for j, data in enumerate(self.trainDataLoader):
                # Every data instance is an input + label pair

                image, mask = data[0], data[1]
                image, mask = image.to(self.device), mask.to(self.device)

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                
                # Make predictions for this batch
                prediction = self.model(image)

                mask = torch.unsqueeze(mask, 1) # use this so our tensor has the shape [16, 64, 64]

                # Compute the loss and its gradients
                loss = self.criterion(prediction, mask)
                loss.backward()
                
                # Adjust learning weights
                self.optimizer.step()
                if j % 100 == 0:
                    print(f"epoch = {i}/{self.epoch-1}, iter = {j}/{len(self.trainDataLoader)}, loss = {loss}")
                
                losses.append(loss.item())
            
            mean_losses.append(np.mean(losses))
            losses.clear()
        
            self.model.eval()
            with torch.no_grad():
                for k, data in enumerate(self.valDataLoader):
                    image, mask = data[0], data[1]
                    image, mask = image.to(self.device), mask.to(self.device)
                    mask = torch.unsqueeze(mask, 1)
                    vprediction = self.model(image).to(self.device)
                    validation_loss = self.criterion(vprediction, mask)
                    validation_losses.append(validation_loss.item())
                    if k % 30 == 0:
                        print(f"epoch = {i}/{self.epoch-1}, iter = {k}/{len(self.valDataLoader)}, validation_loss = {validation_loss}")
                mean_validation_loss.append(np.mean(validation_losses))
                validation_losses.clear()
                   
        # Print learning curves

        plt.figure(figsize=(10,5))
        plt.title("Learning Curve")
        x = [i for i in range(1, self.epoch+1)]
        plt.plot(x, [i for i in mean_losses],label="T", marker=".")
        plt.plot(x, [i for i in mean_validation_loss],label="V", marker=".")
        plt.xticks([i for i in range(0, self.epoch, 5)])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("learning_curve2.svg", format="svg")
        #plt.show()
            
        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')

    # --------------------------------------------------------------------------------
    # DEFINITION OF CUSTOM FUNCTION CREATED FOR THE PROJECT
    # --------------------------------------------------------------------------------        

    # --------------------------------------------------------------------------------
    # THRESHOLDING FUNCTION (sigmoid like)
    # INPUTS: 
    #     - net_output(tensor): tensor 2D pytorch tensor containing the predicted masks
    #     - threshold(float)  : 0.5 float threshold to make it a sigmoid
    # --------------------------------------------------------------------------------        
    def thresholding(self, net_output, threshold): # NEW
        binary_output = torch.where(net_output < threshold, torch.tensor(0.0), torch.tensor(1.0))
        return binary_output
    
    # --------------------------------------------------------------------------------
    # IoU METRIC FUNCTION 
    # INPUTS: 
    #     - pred_mask(tensor): tensor 2D pytorch tensor containing the predicted masks
    #     - target_mask(tensor): tensor 2D pytorch tensor containing the ground truth masks
    #     - epsilon(float)  : to avoid 0 division
    # --------------------------------------------------------------------------------      
    def calculate_iou(self, pred_mask, target_mask, epsilon=1e-10):

        target_mask= torch.unsqueeze(target_mask,1)
        iou_list = []
        for i in range(0,target_mask.shape[0]):
            pred_mask_flat = pred_mask[i,:,:,:].view(-1) # flattens the mask
            target_mask_flat = target_mask[i,:,:,:].view(-1)

        # Calculate intersection and union
            intersection = (pred_mask_flat * target_mask_flat).sum()
            union = pred_mask_flat.sum() + target_mask_flat.sum() - intersection

        # Calculate IoU
            iou = (intersection + epsilon) / (union + epsilon)
            iou_list.append(iou)
    
        return iou_list
    
    # --------------------------------------------------------------------------------
    # PIXEL METRIC FUNCTION 
    # INPUTS: 
    #     - pred_mask(tensor): tensor 2D pytorch tensor containing the predicted masks
    #     - target_mask(tensor): tensor 2D pytorch tensor containing the ground truth masks
    #     - epsilon(float)  : to avoid 0 division
    # --------------------------------------------------------------------------------      
    def pixel_accuracy(self, pred_mask, target_mask, epsilon=1e-10):
    
        target_mask= torch.unsqueeze(target_mask,1)
        accuracy_list = []
        for i in range(0,target_mask.shape[0]):
            pred_mask_flat = pred_mask[i,:,:,:].view(-1) # flattens the mask
            target_mask_flat = target_mask[i,:,:,:].view(-1)
            
            correct_pixels = torch.sum(pred_mask_flat == target_mask_flat)
            total_pixels = len(pred_mask_flat)
            accuracy = (correct_pixels + epsilon) / (total_pixels + epsilon)
            accuracy_list.append(accuracy.item())
    
        return accuracy_list
    
    # --------------------------------------------------------------------------------
    # ENTROPY COMPUTATION FUNCTION 
    # INPUTS: 
    #     - PREDICTION MASKS (tensor): tensor 2D pytorch tensor containing the predicted masks for every image augmentation transform (p_transform)
    # --------------------------------------------------------------------------------      
    def entropy(self, p_vertical, p_horizontal, p_rotate, p_gray, p_randomBright):
        
        entropy_matrix = torch.zeros_like(p_vertical)
        T = 5
        
        for i in np.arange(0, entropy_matrix.shape[1]):
            for j in np.arange(0, entropy_matrix.shape[2]):
                Sum = p_vertical[:,i,j] + p_horizontal[:,i,j] + p_rotate[:,i,j] + p_gray[:,i,j] + p_randomBright[:,i,j]
                p1 = Sum / T
                p0 = ( T - Sum ) / T
                if p1 == 0 or p0 == 0:
                    entropy_matrix[:,i,j] = 0
                else:
                    entropy_matrix[:,i,j] = - ( p0 * np.log(p0) + p1 * np.log(p1) )
       
        return entropy_matrix

    # -------------------------------------------------
    # EVALUATION PROCEDURE (ultra basic implementation)
    # -------------------------------------------------
    def evaluate(self):
        self.model.train(False)
        self.model.eval()
        
        # Qualitative Evaluation 
        allInputs, allPreds, allGT = [], [], []
        IoU = []
        Accuracy = []
       
    
        for i, data in enumerate(self.testDataLoader):
            images, GT, resizedImg, vertical_flip, horizontal_flip, rotateFlip, gaussianNoise, grayImage, brightImage = data
            resizedImg  = resizedImg["image"]
            images      = images.to(self.device)
            predictions = self.model(images)

            vertical_flip, horizontal_flip, rotateFlip = vertical_flip.to(self.device), horizontal_flip.to(self.device), rotateFlip.to(self.device)
            gaussianNoise, grayImage, brightImage = gaussianNoise.to(self.device), grayImage.to(self.device), brightImage.to(self.device)
           
            #inference on the t transform
            p_vertical     = self.model(vertical_flip)
            p_horiztontal  = self.model(horizontal_flip)
            p_rotate       = self.model(rotateFlip)
            p_GNoise       = self.model(gaussianNoise)
            p_gray         = self.model(grayImage)
            p_randomBright = self.model(brightImage)

            #apply sigmoid
            predictions = self.thresholding(predictions, 0.5)      
            p_vertical     = self.thresholding(p_vertical, 0.5) 
            p_horizontal  = self.thresholding(p_horiztontal, 0.5) 
            p_rotate       = self.thresholding(p_rotate, 0.5) 
            p_GNoise       = self.thresholding(p_GNoise, 0.5) 
            p_gray         = self.thresholding(p_gray, 0.5)
            p_randomBright = self.thresholding(p_randomBright, 0.5)
            
            #align the masks
            align_vertical = align_mask( p_vertical.cpu().numpy(), "vertical")
            align_vertical = torch.tensor(align_vertical, dtype=torch.float32)
        
            align_horizontal = align_mask( p_horizontal.cpu().numpy(), "horizontal")
            align_horizontal = torch.tensor(align_horizontal, dtype=torch.float32)
            
            align_rotation = align_mask( p_rotate.cpu().numpy(), "rotation")
            align_rotation = torch.tensor(align_rotation, dtype=torch.float32)
            align_rotation = self.thresholding(align_rotation, 0.5)
            
            #computation of entropy 
            image_nb = 2 #change this number to see the transform of another image in the batch
            entropy_matrix = self.entropy(align_vertical[image_nb], align_horizontal[image_nb], align_rotation[image_nb], p_gray[image_nb], p_randomBright[image_nb])
            #augmentation_plots(resizedImg[image_nb], GT[image_nb], vertical_flip[image_nb], horizontal_flip[image_nb], rotateFlip[image_nb], grayImage[image_nb], brightImage[image_nb], p_vertical[image_nb], p_horizontal[image_nb], p_rotate[image_nb], p_gray[image_nb], p_randomBright[image_nb], align_vertical[image_nb], align_horizontal[image_nb], align_rotation[image_nb], entropy_matrix)
            
            images, predictions = images.to('cpu'), predictions.to('cpu')
            allInputs.extend(resizedImg.data.numpy())
            allPreds.extend(predictions.data.numpy())
            allGT.extend(GT.data.numpy())
            
            


        allInputs = np.array(allInputs)
        allPreds  = np.array(allPreds)
        allGT     = np.array(allGT)
        
        showPredictions(allInputs, allPreds, allGT, self.resultsPath)
        
        # save IoU and accuracy results
        with open('IoU.pkl', 'wb') as f:
            pickle.dump(IoU, f)    
        with open('Accuracy.pkl', 'wb') as f:
            pickle.dump(Accuracy, f)    
        
        # Histogram!!!
        
        # Quantitative Evaluation
        # 
        # histogram IoU
        #
        plt.figure(figsize=(10,5))
        plt.title("Evaluation IoU Histogram")
        plt.hist(IoU, bins = [0,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,1])
        plt.xlabel("IoU")
        plt.ylabel("frequency")
        plt.legend()
        plt.savefig("IoU_hist.svg", format="svg")
        #plt.show()
        #
        # plot accuracy
        # histogram acc
        #
        plt.figure(figsize=(10,5))
        plt.title("Evaluation Accuracy Histogram")
        plt.hist(Accuracy, bins = [0,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,1])
        plt.xlabel("Accuracy")
        plt.ylabel("frequency")
        plt.legend()
        plt.savefig("acc_hist.svg", format="svg")
        #plt.show()

        print(f"IoU Score = {np.mean(IoU)}")
        print(f"Accuracy Score = {np.mean(Accuracy)}")



