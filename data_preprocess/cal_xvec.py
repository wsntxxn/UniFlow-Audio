import wespeaker
import tqdm
from h5py import File
import os
from concurrent.futures import ThreadPoolExecutor
# nohup python data_preprocess/cal_xvec.py  > ./log 2>&1 &
dir_path = '/nas-wulanchabu/jiahao.mei/data/tts/LibriTTS'
# dir_path='/nas-wulanchabu/jiahao.mei/data/tts/LibriTTS/test-other/367/130732'

all_files = []  # 这里应该存放 (dir, file) 的元组
for dir, _, file_names in os.walk(dir_path):
    for file in file_names:
        if file.endswith('.wav'):
            all_files.append((dir, file))  # 直接存储参数元组
print(f'Found {len(all_files)} files')

model = None
def load_model():
    global model
    if model is None:
        model = wespeaker.load_model('english')
        model.set_device('cuda:0')

def extract_xvector(audio_path):
    global model
    load_model()
    return model.extract_embedding(audio_path)

def process_file(file_info):
    dir, file = file_info
    audio_path = os.path.join(dir, file)
    embed = extract_xvector(audio_path)
    return file.replace('.wav', ''), embed

embeds = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(tqdm.tqdm(executor.map(process_file, all_files), total=len(all_files)))
    embeds = dict(results)

h5_save_path = '/nas-wulanchabu/jiahao.mei/code/x2audio/data/libritts/xvector.h5'
with File(h5_save_path, 'w') as hf:
    hf.create_group('xvector')
    for id in embeds.keys():
        hf['xvector'][id] = embeds[id]
print(f'save xvector to {h5_save_path}')
