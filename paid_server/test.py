from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from web3 import Web3
from fastapi.middleware.cors import CORSMiddleware
from snet import sdk
app = FastAPI(debug=True)

# Bật CORS cho phép mọi nguồn gốc truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain (có thể giới hạn cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép mọi phương thức HTTP (POST, GET, etc.)
    allow_headers=["*"],  # Cho phép mọi headers
)

# Kết nối với mạng blockchain (ví dụ: Sepolia thông qua Infura)
infura_url = "https://sepolia.infura.io/v3/0e1cebf226f04d07b89c8bb108aed925"
web3 = Web3(Web3.HTTPProvider(infura_url))
config = sdk.config.Config(
    private_key='1fcb24f4f19f5ca80f645083aaf6d4a9297c00623b1b4c0f502ae89b0d5d6715',
    eth_rpc_endpoint="https://sepolia.infura.io/v3/0e1cebf226f04d07b89c8bb108aed925",
    concurrency=False,
    force_update=False
)
snet_sdk = sdk.SnetSDK(config)

# Cấu hình thông tin dịch vụ
org_id = "111002"
service_id = "lesion_service_1"
group_name = "gr"
# Tạo service client
service_client = snet_sdk.create_service_client(
    org_id=org_id,
    service_id=service_id,
    group_name=group_name
)
result = service_client.call_rpc(
            "StartInfer", 
            "InferRequest", 
            image_path="data/endo/d.jpeg", 
            logId="1", 
            sessionId="1"
        )
print(result)