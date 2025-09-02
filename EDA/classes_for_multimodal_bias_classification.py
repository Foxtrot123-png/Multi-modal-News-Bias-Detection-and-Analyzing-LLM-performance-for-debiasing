import json
from typing import Literal
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchinfo
from transformers import AutoTokenizer, BertModel, AutoModel
from PIL import Image
import requests
from torchvision import transforms
from huggingface_hub import hf_hub_download
from xgboost import XGBClassifier
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataloader_NBS_Plus(Dataset):
    def __init__(self, dataframe, text_tokenizer, image_transform):
        self.df = dataframe
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['article_text']
        image_path = row['image_path']

        # tokenize text
        text_inputs = self.text_tokenizer(
            text, padding='max_length', truncation=True, max_length=512 , return_tensors='pt'
        )
        # transform image
        image = Image.open(image_path).convert('RGB')
        image_input = self.image_transform(image)

        #label = torch.tensor(row['MultiModal_Label'])
        
        input_ids= text_inputs['input_ids'][0]
        attention_mask= text_inputs['attention_mask']
        pixel_values = image_input
        labels =  int(row['MultiModal_Label'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'pixel_values': image_input,
            'labels': torch.tensor(labels, dtype=torch.long)
        }
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
MAX_LEN = 512

class Dataloader_Babe(Dataset):
    def __init__(self, dataframe, tokenizer, MAX_LEN):
        self.tokenizer = tokenizer
        self.text1 = dataframe['article'].values
        self.text2 = dataframe['text'].values
        self.targets = dataframe['label_bias'].values
        self.MAX_LEN = 512

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, index):
        text1 = str(self.text1[index])
        text2 = str(self.text2[index])

        inputs = self.tokenizer(

            text1,
            text2,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    

def train_ldr_for_babe(X_train,batch_size=2):
    
    training_set = Dataloader_Babe(X_train, text_tokenizer, MAX_LEN=512)
    train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }
    
    training_loader = DataLoader(training_set, **train_params)
    return training_loader


def valid_dataloader_nbs(dataset):
    
    test_loader_dataset = Dataloader_NBS_Plus(dataset, text_tokenizer, image_transform)

    val_loader = DataLoader(test_loader_dataset, batch_size=4, shuffle=True)
    return val_loader


def valid_ldr_for_babe(X_test,batch_size=2):
    
    valid_set = Dataloader_Babe(X_test, text_tokenizer, MAX_LEN=512)
    valid_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }
    
    valid_loader = DataLoader(valid_set, **valid_params)
    return valid_loader

class BertClass(torch.nn.Module):
    def __init__(self,drop=0.1):
        super(BertClass, self).__init__()
        #self.l1 = BertModel.from_pretrained("E:\\GitHub\\Multi-Model-Bias-Detection-and-Debiasing-the-News\\EDA\\Model_config")
        self.l1 = BertModel.from_pretrained("/Users/ritikrmohapatra/Documents/GitHub/Multi-Model-Bias-Detection-and-Debiasing-the-News/EDA/Model_config")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(p=drop)
        self.classifier = torch.nn.Linear(768, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class MultimodalClassifier(nn.Module):
    def __init__(
            self,
            text_encoder_id_or_path: str,
            image_encoder_id_or_path: str,
            projection_dim: int,
            proj_dropout: float = 0.1,
            fusion_dropout: float = 0.1,
            num_classes: int = 1,
            num_attention_heads=8
            
        ) -> None:
        super().__init__()

        self.projection_dim = projection_dim
        self.num_classes = num_classes

        self.text_encoder = BertModel.from_pretrained(text_encoder_id_or_path)
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
        )

        self.image_encoder = AutoModel.from_pretrained(image_encoder_id_or_path, trust_remote_code=True)
        self.image_encoder.classifier = nn.Identity()  # rm the classification head
        self.image_projection = nn.Sequential(
            nn.Linear(512, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(self.projection_dim, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        

        ##### Classification Layer
        self.classifier = nn.Linear(self.projection_dim, self.num_classes)
        self.dropout = nn.Dropout(fusion_dropout)


    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        full_text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        full_text_features = self.text_projection(full_text_features)

        image_features = self.image_encoder(pixel_values=pixel_values).last_hidden_state  
        B, C, H, W = image_features.shape
        image_features = image_features.view(B, C, H*W).permute(0, 2, 1)  
        image_features = self.image_projection(image_features) 

        attended_text, _ = self.cross_attention(
            query=full_text_features,   
            key=image_features,    
            value=image_features   
        )

        cls_rep = attended_text[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep)

        return logits

def load_model(drop_proj=0.1,drop_fus=0.1):
    
    config = {
        "model_type": "multimodal-bias-classifier", 
        #"text_encoder_id_or_path": "E:\\GitHub\\Multi-Model-Bias-Detection-and-Debiasing-the-News\\EDA\\Model_config", #For Windows
        "text_encoder_id_or_path": "/Users/ritikrmohapatra/Documents/GitHub/Multi-Model-Bias-Detection-and-Debiasing-the-News/EDA/Model_config",
        "image_encoder_id_or_path": "resnet34", 
        "projection_dim": 768, 
        "num_classes": 1,
        "hidden_size": 768, 
        "save_components": ["resnet_encoder", "text_encoder", "fusion_layer", "classifier"], 
        "exclude_components": ["clip_text_encoder", "clip_image_encoder"]}
    model = MultimodalClassifier(
        text_encoder_id_or_path=config["text_encoder_id_or_path"],
        image_encoder_id_or_path="microsoft/resnet-34",
        projection_dim=config["projection_dim"],
        proj_dropout=drop_proj,
        fusion_dropout=drop_fus,
        num_classes=config["num_classes"]
    )

    model_weights_path = hf_hub_download(repo_id="maximuspowers/multimodal-bias-classifier", filename="model_weights.pth")
    checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu')) #wINDOWS 

    model.load_state_dict(checkpoint, strict=False)

    return model


class EnsembleModel(torch.nn.Module):
    def __init__(self,model_text_only,model_txt_and_img,valid_for_text,valid_for_imgandtxt,valid_loader_txt,valid_loader_textAndImage):
        super().__init__()
        ## Model Initialization
        self.model_text_only = model_text_only
        self.model_txt_and_img = model_txt_and_img

  

        #Dataloaders
        self.valid_loader_textAndImage = valid_loader_textAndImage
        self.valid_loader_txt = valid_loader_txt

        self.valid_for_text = valid_for_text
        self.valid_for_imgandtxt = valid_for_imgandtxt



    def forward(self,input_text,input_img_and_txt):
        probs_text,true = self.valid_for_text(self.model_text_only,self.valid_loader_txt(input_text))
        probs_textAndImage,true_ = self.valid_for_imgandtxt(self.model_txt_and_img,self.valid_loader_textAndImage(input_img_and_txt))
        

        X = np.vstack([probs_textAndImage.reshape(-1), probs_text.reshape(-1)]).T
        y = true

        X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(
            n_estimators=200,        
            learning_rate=0.1,       
            max_depth=3,             
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )


        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        return y_pred_val,y_val
    


from tqdm import tqdm

loss_function = torch.nn.CrossEntropyLoss()



def valid_BABE(model, testing_loader,thres=0.30000000000000004):
    prob_all = []
    prob_target = []
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].reshape(-1,1).to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            preds = torch.sigmoid(outputs) > thres    
            prob = torch.sigmoid(outputs)
            prob_all.append(prob)
            prob_target.append(targets)
            
            
    prob_all = torch.cat(prob_all).numpy()       
    prob_target = torch.cat(prob_target).numpy()
    return prob_all,prob_target

def valid_NBS(model, testing_loader,thres=0.28301114938459293):
    model.eval()
    prob_all = []
    prob_target = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            pixel_values = data['pixel_values'].to(device, dtype = torch.float)
            targets = data['labels'].reshape(-1,1).to(device, dtype = torch.float)

            outputs = model(input_ids=ids,
            attention_mask=mask,
            pixel_values=pixel_values)

            preds = torch.sigmoid(outputs) > thres    
            prob = torch.sigmoid(outputs)
            prob_all.append(prob)
            prob_target.append(targets)
            

    prob_all = torch.cat(prob_all).numpy()       
    prob_target = torch.cat(prob_target).numpy()
    return prob_all,prob_target

#https://stackoverflow.com/questions/62301674/extracting-labels-after-applying-softmax  --> For Probability, Weights and Labels


class EnsembleModel_for_single_pred(torch.nn.Module):
    def __init__(self,model_text_only,model_txt_and_img,valid_for_text,valid_for_imgandtxt,valid_loader_txt,valid_loader_textAndImage):
        super().__init__()
        ## Model Initialization
        self.model_text_only = model_text_only
        self.model_txt_and_img = model_txt_and_img

  

        #Dataloaders
        self.valid_loader_textAndImage = valid_loader_textAndImage
        self.valid_loader_txt = valid_loader_txt

        self.valid_for_text = valid_for_text
        self.valid_for_imgandtxt = valid_for_imgandtxt



    def predict(self,input_text,input_img_and_txt):
        probs_text,true = self.valid_for_text(self.model_text_only,self.valid_loader_txt(input_text))
        probs_text=float(probs_text)
        if input_img_and_txt is not None:
            probs_textAndImage,true_ = self.valid_for_imgandtxt(self.model_txt_and_img,self.valid_loader_textAndImage(input_img_and_txt))
            probs_textAndImage=float(probs_textAndImage)
            comb = round(0.8*probs_text+0.2*probs_textAndImage)>0.5
        else:
            comb=probs_text>0.5
        

        return comb
    

