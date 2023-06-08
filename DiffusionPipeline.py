import torch, logging 
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
logging.disable(logging.WARNING) 

class StableDiffusion():

  def __init__(self):
    self.device = "cuda"
    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    self.encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")

    self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16).to(self.device)

    self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    self.scheduler.set_timesteps(70)

    self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16).to(self.device)

  def load_image(Photo):

      return Image.open(Photo).convert('RGB').resize((512,512))

  def image2latents(self, image):

      init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
      init_image = init_image.to(device=self.device, dtype=torch.float16) 
      init_latent_dist = self.vae.encode(init_image).latent_dist.sample() *  0.19112
      return init_latent_dist

  def latents2image(self, latents_feature_space):
  
      latents_feature_space = (1 / 0.19112) * latents_feature_space
      with torch.no_grad():
          image = self.vae.decode(latents_feature_space).sample
      image = (image / 2 + 0.5).clamp(0, 1)
      image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
      images = (image * 255).round().astype("uint8")
      pil_images = [Image.fromarray(image) for image in images]
      return pil_images

  def textencoding(self, prompts):
      maxlen = self.tokenizer.model_max_length
      inp = self.tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt") 
      return self.encoder(inp.input_ids.to(self.device))[0].half()

  def text2image(self, prompts, guidence_scale=7.5, seed=None, steps=70, dim=512):
      
      batch_size = len(prompts) 
      
      text = self.textencoding(prompts) 
      

      uncondition =  self.textencoding([""] * batch_size, text.shape[1])

      emb = torch.cat([uncondition, text])
      
      if seed: 
          torch.manual_seed(seed)

      latents_feature_space = torch.randn((batch_size, self.unet.config.in_channels, dim//8, dim//8))
      
      self.scheduler.set_timesteps(steps)
      
      latents_feature_space = latents_feature_space.to(self.device).half() * self.scheduler.init_noise_sigma
      
      for i,ts in enumerate(tqdm(self.scheduler.timesteps)):

          inp = self.scheduler.scale_model_input(torch.cat([latents_feature_space] * 2), ts)
          
          with torch.no_grad(): 
              u,t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
              
          pred = u + guidence_scale*(t-u)
          

          latents_feature_space = self.scheduler.step(pred, ts, latents_feature_space).prev_sample
          

      return self.latents2image(latents_feature_space)

  def image2image(self, prompts, init_img, guidence_scale=7.5, seed=None, strength =0.8, steps=70):

      text = self.textencoding(prompts) 

      batch_size = len(prompts) 


      uncondition =  self.textencoding([""] * batch_size, text.shape[1])

      emb = torch.cat([uncondition, text])
      
      if seed: 
          torch.manual_seed(seed)
      
      self.scheduler.set_timesteps(steps)
      
      init_latents = self.image2latents(init_img)
      
      init_timestep = int(steps * strength) 
      timesteps = self.scheduler.timesteps[-init_timestep]
      timesteps = torch.tensor([timesteps], device=self.device)
      
      noise = torch.randn(init_latents.shape, generator=None, device=self.device, dtype=init_latents.dtype)
      init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
      latents_feature_space = init_latents
      
      t_start = max(steps - init_timestep, 0)
      timesteps = self.scheduler.timesteps[t_start:].to(self.device)
      
      for i,ts in enumerate(tqdm(timesteps)):

          inp = self.scheduler.scale_model_input(torch.cat([latents_feature_space] * 2), ts)
          
          with torch.no_grad(): 
              u,t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

          pred = u + guidence_scale*(t-u)
          
          latents_feature_space = self.scheduler.step(pred, ts, latents_feature_space).prev_sample
          

      return self.latents2image(latents_feature_space)