# Very simple and custom script to download and prepare the dataset. 
# It also splits the dataset into training, validation and test sets.

from argparse import ArgumentParser
import gdown, zipfile, os, glob, shutil, random, re
from PIL import Image
# For reproducibility of the splits
random.seed(42)


def prepare_our_dataset():
    # Source https://drive.google.com/file/d/16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa/view?usp=drive_link

    # Occasionaly, the download via gdown in python fails. In that case, just re-try.

    # file_id="16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa"
    # gdown.download(id=f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}")
    # gdown.download(id="16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa", use_cookies=False)

    # Alternative: use the gdown command line tool
    os.system("gdown --no-cookie 16pMYcodc5oshaTIcQYcHrK2VDG7vUuLa")

    # Alternative: manually download the file from the link above and place it in the same folder as this script
    if not os.path.exists("dl_action_dataset.zip"):
        raise FileNotFoundError("Dataset zip file not found. Please download it manually from the link above.")

    with zipfile.ZipFile("dl_action_dataset.zip", 'r') as f:
        f.extractall(".")


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



    print("Resizing the images...")
    # Resize the images
    

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

    print(available_actions)

def prepare_stanford():

    os.system("wget http://vision.stanford.edu/Datasets/Stanford40.zip")

    save_dir="Stanford40"
    os.mkdir(save_dir)
    with zipfile.ZipFile("Stanford40.zip", 'r') as f:
        f.extractall(save_dir)


    processed_dir=os.path.join(save_dir, "processed")
    os.mkdir(processed_dir)

    imgs_dir=os.path.join(save_dir, "JPEGImages")

    # First step: rename the folders and move the images
    available_actions={}

    for img_path in sorted(glob.glob(f"{imgs_dir}/*")):

        img_name=img_path.split("/")[-1]
        action_name="_".join(img_name.split("_")[:-1]) # remove the last part of the name (the number)

        target_path=os.path.join(processed_dir, action_name)

        if action_name not in available_actions.keys():
            available_actions[action_name]=0
            os.mkdir(target_path)


        available_actions[action_name]+=1
        idx=available_actions[action_name]

        shutil.move(img_path, f"{target_path}/{idx:0>5}.jpg")

    # rm unnecessary folders
    shutil.rmtree(os.path.join(save_dir, "MatlabAnnotations"))
    print(available_actions)




if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("-ds", "--dataset", required=True, type=str, help="Dataset name", choices=["our", "Stanford40"])
    args = args.parse_args()

    print("Preparing dataset...")
    if args.dataset == "our":
        prepare_our_dataset()
    elif args.dataset == "Stanford40":
        prepare_stanford()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented") 
    
    print("Done!")