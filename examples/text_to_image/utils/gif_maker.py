from PIL import Image
from pathlib import Path
from tqdm import tqdm
import fnmatch
# from IPython.display import Image as im, HTML
from random import sample
import cv2
import os
from base64 import b64encode

def check_if_multiple_logged_images(key, run):
    for log in run.scan_history(keys=[key]):
        if log[key]['_type'] == 'images/separated':
            return True
        elif log[key]['_type'] == 'image-file':
            return False
        else:
            return False

def get_number_of_images(key, run):
    for log in run.scan_history(keys=[key]):
        return log[key]['count']

def paste_to_bg(x, bg):
    "adds image to background, but since .paste is in-place, we operate on a copy"
    bg2 = bg.copy()
    bg2.paste(x, x)  # paste returns None since it's in-place 
    return bg2 

def images_to_gif(image_fnames, fname, NUM_IMAGES_PER_GIF=86, DURATION=200):
    if not image_fnames: return 
    image_fnames.sort(key=lambda x: int(x.name.split('_')[-2])) #sort by step

    frames = [Image.open(image) for image in image_fnames]
    max_x = 1024    # 1024x GIF is displayable within RAM limit of Colab free version. mp4s can go bigger.
    if frames[0].size[0] > max_x:
        print(f"Rescaling to ({max_x},..) so as not to exceed Colab RAM.")
        ratio = max_x/frames[0].size[0]
        newsize = [int(x*ratio) for x in frames[0].size]
        if newsize[1]%2 != 0: newsize[1] += 1 # wow ffmpeg hates odd dimensions! 
        frames = [x.resize(newsize, resample=Image.BICUBIC) for x in frames]

    if frames[0].mode == 'RGBA':  # transparency goes black when saved as gif, so let's put it on white first
        bg = Image.new('RGBA', frames[0].size, (255, 255, 255)) 
        frames = [paste_to_bg(x,bg).convert('RGB').convert('P', palette=Image.ADAPTIVE) for x in frames]

    print("saving gif")
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
               save_all=True, duration=DURATION, loop=0)
    
    print("making mp4")
    w, h = frames[0].size
    cmd = f"ffmpeg -loglevel error -i {f'{fname}.gif'} -vcodec libx264 -crf 25 -pix_fmt yuv420p {f'{fname}.mp4'}"
    os.system(cmd)
    if not os.path.exists(f'{fname}.mp4'):
        print(f"Failed to create mp4 file: {fname}.mp4\")\n")

def make_gifs(key, run, extension, NUM_IMAGES_PER_GIF=86, DURATION=200):
    if check_if_multiple_logged_images(key, run):
        count = get_number_of_images(key, run)
        for i in range(count):
            image_fnames = list(Path('./media/images/').glob(f'{key}*{i}{extension}'))
            images_to_gif(image_fnames, f'{key}_{i}', NUM_IMAGES_PER_GIF=NUM_IMAGES_PER_GIF, DURATION=DURATION)
    else:
        image_fnames = list(Path('./media/images/').glob(f'{key}*{extension}'))
        images_to_gif(image_fnames, key, NUM_IMAGES_PER_GIF=NUM_IMAGES_PER_GIF, DURATION=DURATION)

def download_files(filenames_to_download, run):
    keys = set()
    print('Downloading Files')
    for file in tqdm(run.files()):
        if Path(file.name).is_file():
            continue
        if Path(file.name).name not in filenames_to_download:
          continue
        file.download()
    return keys

def sample_fnames(matching_fnames, NUM_IMAGES_PER_GIF=86):
  length = len(matching_fnames)
  if length > NUM_IMAGES_PER_GIF:
    matching_fnames.sort(key=lambda x: int(x.split('_')[-2])) #sort by step 
    fnames = sample(matching_fnames, NUM_IMAGES_PER_GIF)
    return fnames
  else:
    return matching_fnames

def get_filenames_for_key(key, all_filenames, extension, run, NUM_IMAGES_PER_GIF=86):
  if check_if_multiple_logged_images(key, run):
    count = get_number_of_images(key, run)
    filenames_for_key = []
    for i in range(count):
      matching_fnames = fnmatch.filter(all_filenames, f'{key}*{i}{extension}')
      filenames_for_key.extend(sample_fnames(matching_fnames, NUM_IMAGES_PER_GIF=NUM_IMAGES_PER_GIF))
    return filenames_for_key
  else: 
    matching_fnames = fnmatch.filter(all_filenames, f'{key}*{extension}')
    length = len(matching_fnames)
    return sample_fnames(matching_fnames, NUM_IMAGES_PER_GIF=NUM_IMAGES_PER_GIF)
  
# def display_gif(path):
#   print(f'Generated gif: {path}')
#   display(im(data=open(path,'rb').read(), format='png'))
 
# def display_mp4(path, video_width = 600):
#   video_file = open(path, "r+b").read()
#   video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
#   print(f'Generated mp4: {path}')
#   display(HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>"""))

def make_and_display_gifs(run, NUM_IMAGES_PER_GIF=86, DURATION=200):
  extension = ".png"
  all_filenames = [Path(file.name).name for file in run.files() if file.name.endswith(extension)]
  keys = set([Path(fname).stem.split('_')[0] for fname in all_filenames]) 
  for key in keys:
    download_files(get_filenames_for_key(key, all_filenames, extension, run, NUM_IMAGES_PER_GIF=NUM_IMAGES_PER_GIF), run)
  for key in keys:
    path = make_gifs(key, run, extension, NUM_IMAGES_PER_GIF=NUM_IMAGES_PER_GIF, DURATION=DURATION)
#   for path in Path('/content/').glob('*.gif'):
#     display_mp4(path.name.replace('.gif', '.mp4', 1))
#     display_gif(path)   # make display_gif call last because this is what crashes Colab Runtime. 