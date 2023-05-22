import os
from torch.utils.data import Dataset
from PIL import Image
class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir ='dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = Mydata(root_dir,ants_label_dir)
bees_dataset = Mydata(root_dir,bees_label_dir)
#因为继承了Dataset类，相加是内置的功能，所以不用再手动修改__add__了
train_dataset = ants_dataset+bees_dataset

root_dir = 'dataset/train'
target_dir = 'ants_image'
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]
out_dir = 'ants_label'
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(label)

# from PIL import Image
# img_path = r"D:\PythonCode\pytorch\dataset\train\ants\0013035.jpg"
# img = Image.open(img_path)
# img = Image.new('RGB',(160,90),(0,0,255))
# img.show()
# #img.show()
# input("Press Enter to continue...")
# from PIL import Image
# import matplotlib.pyplot as plt
#
# img_path = r"D:\PythonCode\pytorch\dataset\train\ants\0013035.jpg"
# img = Image.open(img_path)
# plt.imshow(img)
# plt.show()


