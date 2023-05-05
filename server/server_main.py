import gradio as gr
import cv2
from detector import CocoDetector
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Server for detector inference.")
parser.add_argument("--model_path", type=str, default=str(Path(__file__).parent / "models" / "yolov8n.onnx"))




# interface = gr.Interface(fn=to_black, inputs="image", outputs="image")
# interface.launch()

if __name__ == "__main__":
    args = parser.parse_args()
    coco_detector = CocoDetector(
        model_path=args.model_path, 
    )
    def inference(img):
        detection, img_vis = coco_detector.predict(img)
        return img_vis, "\n".join([str(d) for d in detection])
    interface = gr.Interface(
        fn=inference, 
        inputs="image", 
        outputs=["image", "text"], 
        examples=[[str(Path(__file__).parent / "examples" / "demo.jpg")]]
        )
    interface.launch(
        
    )