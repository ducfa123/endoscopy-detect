from web3 import Web3

# Kết nối tới RPC của mạng (ví dụ: Sepolia Testnet)
web3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/09027f4a13e841d48dbfefc67e7685d5"))

# Transaction hash
tx_hash = "0xb0f719362be3a91a3625d04eb4faabd8cf271ddbd3d6254fa7b5ecd931715d8b"

# Lấy chi tiết giao dịch
try:
    receipt = web3.eth.get_transaction_receipt(tx_hash)
    print("Transaction Receipt:")
    print(receipt)
except Exception as e:
    print(f"Error fetching transaction receipt: {str(e)}")
