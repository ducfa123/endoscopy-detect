import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from web3 import Web3
from fastapi.middleware.cors import CORSMiddleware
from snet import sdk
import json
import shutil

# Load các biến môi trường từ file .env
load_dotenv() 

app = FastAPI(debug=True)
app.mount("/media", StaticFiles(directory="data"), name="media")

# Bật CORS cho phép mọi nguồn gốc truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain (có thể giới hạn cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép mọi phương thức HTTP (POST, GET, etc.)
    allow_headers=["*"],  # Cho phép mọi headers
)

# Cấu hình thông tin dịch vụ
seller_private_key = os.getenv('SELLER_PRIVATE_KEY')
infura_url = os.getenv('INFURA_URL')
eth_rpc_endpoint = os.getenv('ETH_RPC_ENDPOINT')
org_id = os.getenv('ORG_ID')
service_id = os.getenv('SERVICE_ID')
group_name = os.getenv('GROUP_NAME')
# ABI và Bytecode của hợp đồng Escrow
contract_abi = [
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
contract_bytecode = "608060405234801561000f575f80fd5b5060405161077e38038061077e83398181016040528101906100319190610114565b335f806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055508060015f6101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505061013f565b5f80fd5b5f73ffffffffffffffffffffffffffffffffffffffff82169050919050565b5f6100e3826100ba565b9050919050565b6100f3816100d9565b81146100fd575f80fd5b50565b5f8151905061010e816100ea565b92915050565b5f60208284031215610129576101286100b6565b5b5f61013684828501610100565b91505092915050565b6106328061014c5f395ff3fe608060405260043610610049575f3560e01c806308551a531461004d5780637150d8ae1461007757806386d1a69f146100a1578063aa8c217c146100b7578063d0e30db0146100e1575b5f80fd5b348015610058575f80fd5b506100616100eb565b60405161006e91906103a2565b60405180910390f35b348015610082575f80fd5b5061008b610110565b60405161009891906103a2565b60405180910390f35b3480156100ac575f80fd5b506100b5610133565b005b3480156100c2575f80fd5b506100cb610274565b6040516100d891906103d3565b60405180910390f35b6100e961027a565b005b60015f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b5f8054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b5f8054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16146101c0576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016101b790610446565b60405180910390fd5b5f60025411610204576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016101fb906104ae565b60405180910390fd5b60015f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff166108fc60025490811502906040515f60405180830381858888f1935050505015801561026a573d5f803e3d5ffd5b505f600281905550565b60025481565b5f8054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff1614610307576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016102fe90610516565b60405180910390fd5b5f3411610349576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016103409061057e565b60405180910390fd5b3460025f82825461035a91906105c9565b92505081905550565b5f73ffffffffffffffffffffffffffffffffffffffff82169050919050565b5f61038c82610363565b9050919050565b61039c81610382565b82525050565b5f6020820190506103b55f830184610393565b92915050565b5f819050919050565b6103cd816103bb565b82525050565b5f6020820190506103e65f8301846103c4565b92915050565b5f82825260208201905092915050565b7f4f6e6c79207468652062757965722063616e2072656c656173652066756e64735f82015250565b5f6104306020836103ec565b915061043b826103fc565b602082019050919050565b5f6020820190508181035f83015261045d81610424565b9050919050565b7f4e6f2066756e647320617661696c61626c6520746f2072656c656173650000005f82015250565b5f610498601d836103ec565b91506104a382610464565b602082019050919050565b5f6020820190508181035f8301526104c58161048c565b9050919050565b7f4f6e6c79207468652062757965722063616e206465706f7369740000000000005f82015250565b5f610500601a836103ec565b915061050b826104cc565b602082019050919050565b5f6020820190508181035f83015261052d816104f4565b9050919050565b7f4d7573742073656e6420736f6d652065746865720000000000000000000000005f82015250565b5f6105686014836103ec565b915061057382610534565b602082019050919050565b5f6020820190508181035f8301526105958161055c565b9050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f6105d3826103bb565b91506105de836103bb565b92508282019050808211156105f6576105f561059c565b5b9291505056fea26469706673582212202fac9736cc48b7e41aae4a8f75e6dae054f5efa32c51071a0a1081d09f1acb5064736f6c634300081a0033"

web3 = Web3(Web3.HTTPProvider(infura_url))
config = sdk.config.Config(
    private_key=seller_private_key,
    eth_rpc_endpoint=eth_rpc_endpoint,
    concurrency=False,
    force_update=False
)
snet_sdk = sdk.SnetSDK(config)


# Tạo service client
service_client = snet_sdk.create_service_client(
    org_id=org_id,
    service_id=service_id,
    group_name=group_name
)


# Tài khoản triển khai hợp đồng (lấy private key từ MetaMask)
account = web3.eth.account.from_key(seller_private_key)
web3.eth.default_account = account.address

def release_payment(contract_address):
    try:
        # Tạo đối tượng hợp đồng Escrow
        Escrow = web3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Tăng giá gas để tránh lỗi 'replacement transaction underpriced'
        gas_price = web3.eth.gas_price * 1.2
        nonce = web3.eth.get_transaction_count(web3.eth.default_account)
        
        # Tạo giao dịch gọi hàm release
        transaction = Escrow.functions.release().build_transaction({
            'from': web3.eth.default_account,  # Địa chỉ người bán (seller)
            'nonce': nonce,
            'gas': 2000000,  # Giới hạn gas
            'gasPrice': int(gas_price)
        })

        # Ký giao dịch bằng private key của seller
        signed_txn = web3.eth.account.sign_transaction(transaction, private_key=seller_private_key)

        # Gửi giao dịch đã ký lên blockchain
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)

        # Chờ giao dịch được xác nhận
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "status": "success",
            "tx_hash": tx_hash.hex()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
# Mô hình dữ liệu cho yêu cầu predict
class EscrowRequest(BaseModel):
    seller_address: str  # Địa chỉ người bán

@app.post("/predict")
async def predict(
    logId: str = Form(...), 
    sessionId: str = Form(...), 
    contract_address: str = Form(...), 
    file: UploadFile = File(...)
):
    try:
        # Lưu file ảnh vào thư mục 'data'
        upload_dir = 'data/'
        os.makedirs(upload_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        fp = os.path.join(upload_dir, file.filename)
        file_path = os.path.abspath(fp)
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        # Gọi hàm call_rpc để thực hiện dự đoán với đường dẫn của file đã lưu
        result = service_client.call_rpc(
            "StartInfer", 
            "InferRequest", 
            image_path=file_path, 
            logId=logId, 
            sessionId=sessionId
        )

        response_json_str = result.response   # Truy cập thuộc tính 'response'

        # Phân tích chuỗi JSON response
        parsed_result = json.loads(response_json_str)

        # Thực hiện thanh toán qua blockchain
        release_result = release_payment(contract_address)
        print(release_result)
        
        if release_result["status"] != "success":
            raise Exception(f"Payment release failed: {release_result['message']}")

        # 1. Lấy đường dẫn ảnh endoscopy từ thư mục output_endoscopy
        endoscopy_dir = os.path.join(upload_dir, 'output_endoscopy')
        endoscopy_images = sorted([f"http://localhost:8000/media/output_endoscopy/{file}" for file in os.listdir(endoscopy_dir) if file.endswith(".png") and not "lesion" in file])

        # 2. Lấy đường dẫn ảnh tổn thương (lesion) từ các thư mục con tương ứng
        lesion_list_url = []
        for i in range(len(endoscopy_images)):
            lesion_folder = os.path.join(endoscopy_dir, f"output_output_endoscopy_image_{i}_lesion")
            if os.path.exists(lesion_folder):
                lesion_images = sorted([f"http://localhost:8000/media/output_endoscopy/output_output_endoscopy_image_{i}_lesion/{file}" for file in os.listdir(lesion_folder) if file.endswith(".png")])
                lesion_list_url.append(lesion_images)
            else:
                # Nếu không có ảnh tổn thương cho ảnh endoscopy này, thêm danh sách rỗng
                lesion_list_url.append([])

        img_path_url = f"http://localhost:8000/media/{file.filename}"

        # Trả về kết quả phân tích cho client dưới dạng JSON với đường dẫn URL hợp lệ
        return {
            "img_path": img_path_url,
            "endoscopy_img_list_path": endoscopy_images,
            "lesion_list_path": lesion_list_url,
            "logId": logId,
            "sessionId": sessionId
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
@app.post("/create-contract")
async def create_contract(request: EscrowRequest):
    try:
        # Chuyển đổi địa chỉ người bán thành checksum address
        checksum_address = web3.to_checksum_address(request.seller_address)
        
        # Tạo đối tượng hợp đồng với ABI và bytecode
        Escrow = web3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
        
        # Tăng giá gas để tránh lỗi 'replacement transaction underpriced'
        gas_price = web3.eth.gas_price * 1.2  
        nonce = web3.eth.get_transaction_count(account.address)
        
        # Tạo giao dịch triển khai hợp đồng
        transaction = Escrow.constructor(checksum_address).build_transaction({
            'from': account.address,
            'nonce': nonce,
            'gas': 100000,  # Giới hạn gas
            'gasPrice': int(gas_price)  # Giá gas đã tăng
        })

        # Ký giao dịch
        signed_txn = account.sign_transaction(transaction)

        # Gửi giao dịch đã ký
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)

        # Chờ giao dịch được xác nhận và lấy địa chỉ hợp đồng
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = tx_receipt.contractAddress

        # Trả về địa chỉ hợp đồng và số tiền cần thanh toán
        value_in_wei = web3.to_wei(0.0000001, 'ether')  # 0.0001 ETH được chuyển thành Wei
        return {
            'contractAddress': contract_address,
            'amount': value_in_wei  # Số tiền trả về ở đơn vị Wei
        }

    except Exception as e:
        # In ra chi tiết lỗi để debug
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
