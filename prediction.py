import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn.functional as F
from base_model import CVCModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVCModel(False)
model.load_state_dict(torch.load("best_accuracy.ckpt",'cpu',weights_only=True)['state_dict'])


file = input("Enter the file path: ")

im = cv2.imread(file)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

transform = A.Compose([
    A.Resize(244,244),
    A.ToFloat(max_value = 255.0),
    ToTensorV2(),
])

batched_im : torch.Tensor = transform(image = im)["image"]
batched_im = batched_im.unsqueeze(0)

key = {
    0:"Angioectasia",
    1:"Bleeding",
    2:"Erosion",
    3:"Erythema",
    4:"Foreign Body",
    5:"Lymphangiectasia",
    6:"Normal",
    7:"Polyp",
    8:"Ulcer",
    9:"Worms"
}

model.eval()
with torch.no_grad():
    output = model(batched_im)
    output = F.softmax(output, dim = -1)
    probab = output.max().item()
    disease = torch.argmax(output, dim=-1).item()
    
print('disease: ',key[disease],'with probability:',probab) #type:ignore