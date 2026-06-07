!pip install -q gradio pillow numpy requests huggingface_hub

import gradio as gr
from PIL import Image
import numpy as np
import tempfile
import os
from gradio_client import Client, handle_file

def run_real_vton(person_img, garment_img, description):
    if person_img is None or garment_img is None:
        return None, "Please upload both images"
    
    person_pil = Image.fromarray(person_img.astype('uint8'))
    garment_pil = Image.fromarray(garment_img.astype('uint8'))
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f1:
        person_path = f1.name
        person_pil.save(person_path)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f2:
        garment_path = f2.name
        garment_pil.save(garment_path)
    
    try:
        client = Client("yisol/IDM-VTON")
        
        result = client.predict(
            dict={
                "background": handle_file(person_path),
                "layers": [],
                "composite": None
            },
            garm_img=handle_file(garment_path),
            garment_des=description,
            is_checked=True,
            is_checked_crop=True,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )
        
        if result and len(result) > 0:
            if isinstance(result[0], str):
                result_img = Image.open(result[0])
            elif isinstance(result[0], dict) and "image" in result[0]:
                result_img = Image.open(result[0]["image"])
            else:
                result_img = result[0]
            
            os.unlink(person_path)
            os.unlink(garment_path)
            
            return np.array(result_img), "Success! Virtual try-on completed"
    except Exception as e:
        pass
    
    os.unlink(person_path)
    os.unlink(garment_path)
    
    return None, f"Error: {str(e)}"

with gr.Blocks(title="VPE-1.0 Virtual Try-On") as demo:
    gr.Markdown("""
    # VPE-1.0: Virtual Try-On Engine
    ## Production-Grade Virtual Try-On with IDM-VTON
    """)
    
    with gr.Row():
        with gr.Column():
            person_input = gr.Image(label="Person Photo", type="numpy", height=350)
        with gr.Column():
            garment_input = gr.Image(label="Garment Photo", type="numpy", height=350)
        with gr.Column():
            output_img = gr.Image(label="Result", type="numpy", height=350)
    
    desc_input = gr.Textbox(label="Garment Description", value="a casual t-shirt")
    status_text = gr.Textbox(label="Status", interactive=False)
    
    btn = gr.Button("Start Virtual Try-On", variant="primary", size="lg")
    
    btn.click(
        fn=run_real_vton,
        inputs=[person_input, garment_input, desc_input],
        outputs=[output_img, status_text]
    )

demo.launch(share=True)
