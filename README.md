# 3D-Image-Reconsruction

As a project requirement for ECE 544 (Pattern Recognition) at UIUC, me and my group implemented a paper published 
by our professor entitled "Occupancy Planes for Single-view RGB-D Human Reconstruction". The link to the paper is https://arxiv.org/abs/2208.02817.

We created the Occupancy Planes by putting together 4 networks: a) Feature Pyramid Network; b) Spatial Network;
c) RGB Network; and d) Depth Network. All of these models are found in modules.py.

My main contribution in this project is writing the dataloader, which includes writing a .csv file that contains the path for each image and
all the pre-processing steps before feeding the images into the networks (dataloader.py). In addition, I also wrote the loss logging and visualization.

The complete end-to-end process can be found in combined_code.ipynb.

We used Jupyter Notebook and VSCode for development and initially trained on Colab. However, due to limited GPU access on Colab,
I eventually trained our model on Kaggle.
