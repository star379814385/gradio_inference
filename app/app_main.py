from gradio_client import Client

if __name__ == "__main__":
    url = r"http://127.0.0.1:7861"
    demo_path = r"D:\code\inference_web\app\data\demo.jpg"
    client = Client(url)
    job = client.submit(demo_path)
    result = job.result()
    print(result)