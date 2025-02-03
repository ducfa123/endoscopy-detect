import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import ipfshttpclient
from web3 import Web3
import json
import grpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from service import service_pb2
from service import service_pb2_grpc
import logging

# Load environment variables from .env
load_dotenv()

app = FastAPI(debug=True)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
seller_private_key = os.getenv('SELLER_PRIVATE_KEY')
infura_url = os.getenv('INFURA_URL')
eth_rpc_endpoint = os.getenv('ETH_RPC_ENDPOINT')
org_id = os.getenv('ORG_ID')
service_id = os.getenv('SERVICE_ID')
group_name = os.getenv('GROUP_NAME')
escrow_contract_address = os.getenv('ESCROW_CONTRACT_ADDRESS')
contract_abi = [
    {
      "inputs": [],
      "name": "deposit",
      "outputs": [],
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "release",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_seller",
          "type": "address"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "inputs": [],
      "name": "amount",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "buyer",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "seller",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    }
  ]

web3 = Web3(Web3.HTTPProvider(infura_url))

try:
    ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to connect to IPFS: {str(e)}")

# Initialize SNet SDK
# from snet import sdk
# config = sdk.config.Config(
#     private_key=seller_private_key,
#     eth_rpc_endpoint=eth_rpc_endpoint,
#     concurrency=False,
#     force_update=False
# )
# snet_sdk = sdk.SnetSDK(config)

# service_client = sdk.SnetSDK(config).create_service_client(
#     org_id=org_id,
#     service_id=service_id,
#     group_name=group_name
# )

def call_grpc_service(image_path: str, log_id: str, session_id: str) -> dict:
    """
    Gọi dịch vụ gRPC và trả về kết quả dưới dạng dictionary.
    
    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        log_id (str): Log ID cho yêu cầu.
        session_id (str): Session ID cho yêu cầu.

    Returns:
        dict: Kết quả trả về từ dịch vụ gRPC.
    """
    try:
        with grpc.insecure_channel('localhost:7777') as channel:
            stub = service_pb2_grpc.InferServiceStub(channel)
            
            # Tạo yêu cầu gRPC
            request = service_pb2.InferRequest(
                image_path=image_path,
                logId=log_id,
                sessionId=session_id
            )
            
            # Gọi dịch vụ gRPC
            response = stub.StartInfer(request)
            
            # Phân tích kết quả
            return {
                "response": json.loads(response.response),
                "logId": response.logId,
                "sessionId": response.sessionId
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gRPC service call failed: {str(e)}")


# Function to upload file to IPFS
def upload_to_ipfs(file_path):
    try:
        res = ipfs_client.add(file_path)
        return res['Hash']  # Return IPFS CID
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPFS upload failed: {str(e)}")

# Function to download file from IPFS
def download_from_ipfs(cid, output_path):
    try:
        with open(output_path, "wb") as file:
            file.write(ipfs_client.cat(cid))  # Retrieve the file content directly
        print(f"Image downloaded from IPFS and saved to {output_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPFS download failed: {str(e)}")
# Function to release payment

def release_payment():
    try:
        account_address = Web3.to_checksum_address(web3.eth.account.from_key(seller_private_key).address)
        web3.eth.default_account = account_address
        escrow_contract = web3.eth.contract(
            address=escrow_contract_address,
            abi=contract_abi
        )

        # Kiểm tra số dư trong contract
        contract_balance = web3.eth.get_balance(escrow_contract.address)
        if contract_balance == 0:
            raise Exception("No funds available in the contract.")

        # Lấy gas và nonce
        gas_limit = escrow_contract.functions.release().estimate_gas({'from': account_address})
        gas_price = int(web3.eth.gas_price * 1.5)
        nonce = web3.eth.get_transaction_count(account_address, 'pending')

        # Xây dựng và ký giao dịch
        txn = escrow_contract.functions.release().build_transaction({
            'from': account_address,
            'nonce': nonce,
            'gas': gas_limit,
            'gasPrice': gas_price
        })
        signed_txn = web3.eth.account.sign_transaction(txn, private_key=seller_private_key)

        # Gửi giao dịch
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        print("Release transaction receipt:", receipt)
        return {"status": "success", "tx_hash": tx_hash.hex()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment release failed: {str(e)}")

@app.post("/predict")
async def predict(input_cid: str = Form(...)):
    try:
        input_file_path = "/media/bui-minh-duc/DATA1/meai-core/paid_server/temp_input_image.jpg"
        # Download file từ IPFS
        download_from_ipfs(input_cid, input_file_path)

        # Gọi dịch vụ gRPC qua hàm tách riêng
        print("Calling gRPC service directly...")
        log_id = "log_example"
        session_id = "session_example"
        grpc_result = call_grpc_service(input_file_path, log_id, session_id)

        # Kết quả từ gRPC
        prediction_result = grpc_result["response"]
        print("Prediction result:", prediction_result)

        # Xử lý lưu kết quả lên IPFS
        endoscopy_cids = []
        lesion_cids = []

        # Upload từng ảnh trong danh sách endoscopy_img_list_path
        for endoscopy_img_path in prediction_result.get("endoscopy_img_list_path", []):
            cid = upload_to_ipfs(endoscopy_img_path)
            endoscopy_cids.append(cid)

        # Upload từng ảnh trong danh sách lesion_list_path
        for lesion_img_path in prediction_result.get("lesion_list_path", []):
            cid = upload_to_ipfs(lesion_img_path)
            lesion_cids.append(cid)

        # Cleanup file đầu vào
        os.remove(input_file_path)

        # Giải phóng thanh toán và kiểm tra
        release_result = release_payment()
        if release_result.get("status") == "success":
          return {
            "status": "success",
            "endoscopy_cids": endoscopy_cids,
            "lesion_cids": lesion_cids,
            "tx_hash": release_result["tx_hash"]
          }
        else:
            raise HTTPException(status_code=500, detail="Payment release failed")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
