
This is a project on using using Multimodal Model for Newss Bias Detection with studying the debaising capabilities of LLMs.
The files descriptions are as follows:
===================================Files==========================================

==================Datasets======================

--------------BABE Dataset :----------------------
This is a dataset conataining articles text and bias labels that re done by a group of experts . It is publicly avaliable on Kraggle

-------------Good News Dataset:----------------------------
It is a dataset that was retrived by the authors of Good News, Everyone! Context Driven Entity-Aware Captioning for News Images used . 
It was publicly avaliable from New York Times API Non Commercial License
It has unlabelled images and news articles related to it

---------------NewBiasDataset------------------
This is a dataset  avaliable on Zenodo 
It has images texts with their respective labels .

======================EDA==========================

----------GoodNews_Dataset.IPYNB-------------
It contains the program for pre training the BERT transformer on the good news dataset and are saved .

-------------Model_config----------
This is where the checkpoints and the weights of the pre trained BERT model is save .

------------Babe_Dataset.IPYNB---------------
This is the text only model using BERT.
This file containes the EDA , Traning , Validation and the Parameter optimization of model.
•	Accuracy: 75.37%
•	Precision (Class 1 – Biased, Class 0 – Non-Biased): 0.83
•	Recall (Class 1 - Biased, Class 0 – Non-Biased): 0.82
•	F1-Score (Class 1 - Biased, Class 0 – Non-Biased): 0.82

-----------BABE_fine_tuned_mdoel.pt--------------
It is where the BABE mdoel(Text Only Model ) is saved .

------------News_Media_Dataset.IPYNB-----------

This is the Multimodal  model using BERT adn ResNet-34.
It uses pre defined weights from a library commented in the code to prevent random initilization of the weightes of the fused model
It uses cross attention for fusion
This file containes the EDA , Traning , Validation and the Parameter optimization of model.
•	Accuracy: 63.85%
•	Precision (Class 1 - Biased, Class 0 – Non-Biased): 0.705
•	Recall (Class 1 - Biased, Class 0 – Non-Biased): 0.597
•	F1-Score (Class 1 - Biased, Class 0 – Non-Biased): 0.65
