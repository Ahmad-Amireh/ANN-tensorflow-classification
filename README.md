# ANN-tensorflow-classification
This is a simple project to implement Artificial Neural Network using TensorFlow 

## Project Statment 
- We need to classify whether a client will churn or not—in other words, whether the customer will leave the bank or stay—based on the provided data.

## Workflow 
1. Create env (conda create -p python=3.12.9) and requirment.txt  
1. Ingest Data (Here data is availble locally)
2. Basic Feature Engineering (Encoding + Scaling)
3. Build Arch for NN (Input ...< 64 ...< 32.. 1 )
4. Using Drop Out (prevent OF)
5. Optimizer for upating learnable parameter (weights, bias)
6. Train model 
7. picke the model (package it)
8. Streamlit to create web app 
9. deploy it on stramlit cloud 

### Note:  trainable parameter (# of Input neuron  * # of Hidden(out) neuron + # of Hidden(out) neuron (bias))


## Tensorflow Workfloe
- Sequential mode (ANN)
- Dense (class to implenet hidden layer)
- Activation func (sigmoid/softmax/tansh/relu/leaky relu)
- Optimizer (Back propogation(update weight))
- loss func 
- metric (accuracy,precision...)
- Tensorboard .. to store Training info inside log files and display it 
  

https://ann-tensorflow-classification-ahmadamireh.streamlit.app/


'conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6.0'
