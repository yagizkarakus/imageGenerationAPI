from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import uvicorn
from io import BytesIO
import base64
from pyngrok import ngrok
import nest_asyncio
from PIL import Image
from DiffusionPipeline import StableDiffusion
from StyleTransfer import StyleTransfer
from Schemas import Images, Image2Image

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],  # İzin verilen frontend etki alanını buraya ekleyin
    allow_methods=["*"],  # İzin verilen HTTP yöntemlerini buraya ekleyin (örneğin GET, POST, vb.)
    allow_headers=["*"],  # İzin verilen başlık alanlarını buraya ekleyin
)


stabledif = StableDiffusion()

NST = StyleTransfer()

device = ('cuda')

@app.get("/")
async def generate(prompt: str, scale: float):
    try:

        image = stabledif.text2image(prompts = [prompt], g=scale)[0]

    
        image.save("testimage.png")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())

        return Response(content=imgstr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/image2image")
async def img2img(data: Image2Image):
    try:

        img = BytesIO(base64.b64decode(str(data.image)))
        img = Image.open(img).convert("RGB").resize((512,512))


        image = stabledif.image2image(prompts = [data.prompt], g=data.scale, init_img = img)[0]
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        imgstr = base64.b64encode(img_bytes).decode("utf-8")

        return {"data":str(imgstr)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/style-transfer")
async def styletransfer(image: Images):
    try:
        content_image = base64.urlsafe_b64encode(base64.b64decode(image.content_image))
        style_image = base64.urlsafe_b64encode(base64.b64decode(image.style_image))
        
        stylized_image = NST.neuralstyle(content_image, style_image)
        
        # Convert the stylized image to bytes and base64 encode it
        buffer = BytesIO()
        stylized_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        imgstr = base64.b64encode(img_bytes).decode("utf-8")
        
        return {"data": str(imgstr)}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))



ngrok_tunnel = ngrok.connect(8000)
public_url = ngrok_tunnel.public_url

print('Public URL: ', public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
