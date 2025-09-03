
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
