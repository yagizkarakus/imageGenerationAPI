from pydantic import BaseModel

class Images(BaseModel):
    content_image: str
    style_image: str
    

class Image2Image(BaseModel):
    image: str
    prompt: str
    scale: float