# Very simple and custom script to download and prepare the dataset. 
# It also splits the dataset into training, validation and test sets.

import gdown, zipfile, os, glob, shutil, random, re

# For reproducibility of the splits
random.seed(42)

print("Preparing dataset...")
# Source https://drive.google.com/file/d/16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa/view?usp=drive_link
# Occasionaly, the download fails. In that case, just re-try.
# file_id="16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa"
# gdown.download(id=f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}")
# gdown.download(id="16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa", use_cookies=False)

# If this fails, try in the terminal:
# gdown --no-cookie 16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa


with zipfile.ZipFile("dl_action_dataset.zip", 'r') as f:
    f.extractall(".")

# os.remove("temp_data.zip")
print("Dataset downloaded and extracted")

# First step: rename the folders and move the images
source_ds="dl_action_dataset"
available_actions={}
idx=0
for video_path in glob.glob(f"{source_ds}/*"):
    # print(".", end="", flush=True)
    video_name=video_path.split("/")[-1]
    
    action_name_id=video_name.split("_")[0]
    # action_name=action_name_id[:-1]
    action_name=re.sub(r'[0-9]', '', action_name_id)

    os.rename(video_path, os.path.join(source_ds, action_name_id))

    video_path=os.path.join(source_ds, action_name_id)
    
    for img in glob.glob(f"{video_path}/rgb/*"):
        shutil.move(img, video_path)
    

    # Remove the unnecessary folders and files
    shutil.rmtree(os.path.join(video_path, "rgb"))
    shutil.rmtree(os.path.join(video_path, "depth"))
    os.remove(os.path.join(video_path, "rgb.txt"))
    os.remove(os.path.join(video_path, "depth.txt"))

    target_path=os.path.join(source_ds, action_name)

    if action_name not in available_actions.keys():
        available_actions[action_name]=0
        os.mkdir(target_path)
        idx=0

    for img in glob.glob(f"{video_path}/*"):
        idx+=1
        shutil.move(img, f"{target_path}/{idx:0>5}.png")
        available_actions[action_name]+=1

    os.rmdir(video_path)


# Resize the images
from PIL import Image

# Original size: 360x640
# Target size: 336x336 (224*1.5=336)

for action in glob.glob(f"{source_ds}/*"):
    action_name=action.split("/")[-1]
    for img in glob.glob(f"{action}/*"):
        im = Image.open(img)
        # crop the image
        im = im.crop((152, 0, 152+336, 336)) # (left, top, right, bottom)
        # save it back
        im.save(img)



# # Second step: split the dataset
# split_ratio=[0.7, 0.15, 0.15]

# for action in glob.glob(f"{source_ds}/*"):
#     action_name=action.split("/")[-1]
    
#     train_path=os.path.join(source_ds, "train", action_name)
#     val_path=os.path.join(source_ds, "val", action_name)
#     test_path=os.path.join(source_ds, "test", action_name)

#     os.makedirs(train_path, exist_ok=True)
#     os.makedirs(val_path, exist_ok=True)
#     os.makedirs(test_path, exist_ok=True)

#     imgs=glob.glob(f"{action}/*")
#     random.shuffle(imgs)

#     train_split=int(len(imgs)*split_ratio[0])
#     val_split=int(len(imgs)*split_ratio[1])
#     test_split=int(len(imgs)*split_ratio[2])

#     for img in imgs[:train_split]:
#         shutil.move(img, train_path)

#     for img in imgs[train_split:train_split+val_split]:
#         shutil.move(img, val_path)

#     for img in imgs[train_split+val_split:]:
#         shutil.move(img, test_path)

#     shutil.rmtree(action)
# print("Dataset splitted")


# Analyze the dataset

from cleanvision import Imagelab

# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(data_path="dl_action_dataset/")
imagelab.find_issues()
imagelab.report()


print("Done!")