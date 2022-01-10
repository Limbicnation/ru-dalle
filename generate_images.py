from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.utils import seed_everything
from deep_translator import GoogleTranslator # pip install deep-translator
import os

# What are you creating

text = 'The Entity from the 4th dimension'
#image_save_dir = '/home/ws-ml/ML-Images/ru-dalle/pics'
image_prefix = 'ugliest-mole'
start_seed = 32124713
images_per_res = 5 #Total number of Images generated
seed_batches = 1
batch_size = 8
upscale_multiplier = '2x'

#Cache directorrytt[]
cache_dir = '/home/ws-ml/ML-Images/ru-dalle/cache'

# run
translated = GoogleTranslator(source='auto', target='ru').translate(text)

#if not os.path.exists(image_save_dir):
#    os.mkdir(image_save_dir) 

# prepare models:
device = 'cuda'
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device, cache_dir=cache_dir)
realesrgan = get_realesrgan('x2', device=device) # x2/x4/x8
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)  # for stable generations you should use dwt=False
ruclip, ruclip_processor = get_ruclip('ruclip-vit-base-patch32-v5')
ruclip = ruclip.to(device)

# Text to generate Images
text = 'лицо киберборга'
#text = 'Портрет красивых женщин' 


#Loop
for seed in range (start_seed, (start_seed+seed_batches)):
    print(f'Your text: {text}')
    print(f'Russian: {translated}')
    print(f'Seed: {seed}')

    seed_everything(seed, deterministic=False)
    pil_images = []
    scores = []
    
    for top_k, top_p, images_num in [
        (2048, 0.995, images_per_res),
        (1536, 0.99, images_per_res),
        (1024, 0.99, images_per_res),
        (1024, 0.98, images_per_res),
        (512, 0.97, images_per_res),
        (384, 0.96, images_per_res),
        (256, 0.96, images_per_res),
        (128, 0.96, images_per_res),
        
    ]:_pil_images, _scores = generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, top_p=top_p)
    pil_images += _pil_images
    scores += _scores

    show(pil_images, 6)

    # Upscale

    sr_images = super_resolution(pil_images, realesrgan)

    # Save each frame

    for i, img in enumerate(sr_images):
        img.save(f'{image_save_dir}-{image_prefix}-{seed}-{i}.jpg')

    print('image batch saved')    

#show(pil_images, 6)
#show(top_images, 6)