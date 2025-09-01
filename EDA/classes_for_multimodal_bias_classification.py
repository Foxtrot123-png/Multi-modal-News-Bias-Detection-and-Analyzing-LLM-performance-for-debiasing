from torch.utils.data import Dataset, DataLoader
import torch
class Dataloader_Babe(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text1 = dataframe['article'].values
        self.text2 = dataframe['text'].values
        self.targets = dataframe['label_bias'].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, index):
        text1 = str(self.text1[index])
        text2 = str(self.text2[index])

        inputs = self.tokenizer(

            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
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
    
    training_set = Dataloader_Babe(X_train, text_tokenizer, MAX_LEN)
    train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }
    
    training_loader = DataLoader(training_set, **train_params)
    return training_loader


def valid_ldr_for_babe(X_test,batch_size=2):
    
    valid_set = Dataloader_Babe(X_test, text_tokenizer, MAX_LEN)
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
