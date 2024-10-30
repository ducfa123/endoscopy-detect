import grpc
from service  import service_pb2
from service import service_pb2_grpc

def run_infer(image_path, log_id, session_id):
    # Create a gRPC channel to connect to the server
    with grpc.insecure_channel('localhost:7777') as channel:
        stub = service_pb2_grpc.InferServiceStub(channel)
        
        # Create a request message
        request = service_pb2.InferRequest(
            image_path=image_path,
            logId=log_id,
            sessionId=session_id
        )
        
        # Make the gRPC call
        response = stub.StartInfer(request)
        
        print("Inference result:", response.response)
        print("Log ID:", response.logId)
        print("Session ID:", response.sessionId)

if __name__ == "__main__":
    run_infer("10.jpg", "log123", "session456")
