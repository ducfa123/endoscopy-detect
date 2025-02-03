from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from snet import sdk
import os
import json
import time
import shutil
from dotenv import load_dotenv

load_dotenv()
service_client = None

config = sdk.config.Config(
    private_key=os.getenv("PRIVATE_KEY"),
    eth_rpc_endpoint=os.getenv("ETH_RPC_ENDPOINT"),
    concurrency=False,
    force_update=False
)

snet_sdk = sdk.SnetSDK(config)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    os.makedirs(os.path.join(STATIC_DIR, "uploads"))
    os.makedirs(os.path.join(STATIC_DIR, "outputs"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_service_client():
    """Get or create service client without opening new channel"""
    global service_client
    if service_client is None:
   
        service_client = snet_sdk.create_service_client(
            org_id="meai3",
            service_id="meai-sv",
            group_name="default_group"
        )
        
    return service_client

def mock_call_service(image_path):
    timestamp = int(time.time())
    
    output_dir = os.path.join(STATIC_DIR, f"outputs/output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lesions"), exist_ok=True)
    
    base_url = os.getenv("BASE_URL")
    
    filename = os.path.basename(image_path)
    static_uploaded_path = f"{base_url}/static/uploads/{filename}"
    shutil.copy2(image_path, os.path.join(STATIC_DIR, f"uploads/{filename}"))
    
    # Ảnh nội soi
    endoscopy_images = []
    for i in range(6):
        endoscopy_path = os.path.join(output_dir, f"output_endoscopy_image_{i}.png")
        shutil.copy2(image_path, endoscopy_path)
        endoscopy_images.append(f"{base_url}/static/outputs/output_{timestamp}/output_endoscopy_image_{i}.png")
    
    # Ảnh bệnh thay đổi
    lesion_images = []
    for i in [0, 1, 2, 5]:
        lesion_path = os.path.join(output_dir, f"lesions/output_output_endoscopy_image_{i}_lesion_0.png")
        shutil.copy2(image_path, lesion_path)
        lesion_images.append(f"{base_url}/static/outputs/output_{timestamp}/lesions/output_output_endoscopy_image_{i}_lesion_0.png")
    
    return {
        "status": "success",
        "response": {
            "img_path": static_uploaded_path,
            "endoscopy_img_list_path": endoscopy_images,
            "lesion_list_path": lesion_images,
        },
        "logId": f"direct_{timestamp}",
        "sessionId": f"session_{timestamp}"
    }
    
@app.get("/")
async def root():
    return {"message": "Welcome to MEAI Mock API"}

@app.post("/call_service")
async def call_service(image: UploadFile):
    timestamp = int(time.time())
    try:
        print(f"\n=== Upload Request ===")
        print(f"Filename: {image.filename}")
        print(f"Content-type: {image.content_type}")

        os.makedirs("logDir", exist_ok=True)
        file_path = os.path.join("logDir", image.filename)
        
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        print(f"\n=== File saved ===")
        print(f"Path: {file_path}")
        print(f"Size: {len(content)} bytes")
       
        try:
            client = get_service_client()

            service_response = client.call_rpc(
                "StartInfer",
                "InferRequest",
                image_path=file_path,
                logId=f"direct_{timestamp}",
                sessionId=f"session_{timestamp}",
            )

            print(f"\n=== Service Response ===")
            print(f"Response: {service_response}")

            response_data = json.loads(service_response.response)
            modified_response = modify_paths_in_response(response_data, timestamp)
            
            return JSONResponse(content={
                "status": "success",
                "response": {
                    **modified_response,
                    "logId": service_response.logId,
                    "sessionId": service_response.sessionId
                }
            })

        except Exception as e:
            print(f"Error calling service: {str(e)}")
            print(f"Error details: {type(e).__name__} - {str(e)}")
            raise e
        

        # Use mock service by default
        # mock_response = mock_call_service(file_path)
        
        # print(f"\n=== Mock Service Response ===")
        # print(json.dumps(mock_response, indent=2))
        
        # return JSONResponse(content=mock_response)

    except Exception as e:
        print(f"\n=== Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "type": type(e).__name__
            },
            status_code=500
        )

def modify_paths_in_response(response_data, timestamp):
    """Thay đổi đường dẫn trong response và copy files vào static"""
    modified_data = response_data.copy()
    
    output_dir = os.path.join(STATIC_DIR, "outputs", f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    if 'img_path' in modified_data:
        src_path = modified_data['img_path']
        filename = os.path.basename(src_path)
        static_path = os.path.join(STATIC_DIR, "uploads", filename)
        shutil.copy2(src_path, static_path)
        modified_data['img_path'] = f'/static/uploads/{filename}'
    
    if 'endoscopy_img_list_path' in modified_data:
        new_paths = []
        for path in modified_data['endoscopy_img_list_path']:
            if os.path.exists(path):
                filename = os.path.basename(path)
                static_path = os.path.join(output_dir, filename)
                shutil.copy2(path, static_path)
                new_paths.append(f'/static/outputs/output_{timestamp}/{filename}')
        modified_data['endoscopy_img_list_path'] = new_paths
    
    if 'lesion_list_path' in modified_data:
        new_paths = []
        for path in modified_data['lesion_list_path']:
            if os.path.exists(path):
                filename = os.path.basename(path)
                lesion_dir = os.path.join(output_dir, "lesions")
                os.makedirs(lesion_dir, exist_ok=True)
                static_path = os.path.join(lesion_dir, filename)
                shutil.copy2(path, static_path)
                new_paths.append(f'/static/outputs/output_{timestamp}/lesions/{filename}')
        modified_data['lesion_list_path'] = new_paths
    
    return modified_data
