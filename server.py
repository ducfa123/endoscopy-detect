import grpc
import asyncio
import json
import time

# Import generated classes
from service import service_pb2
from service import service_pb2_grpc

from infer import infer

# Define the gRPC server class inherited from the generated Servicer
class InferServiceServicer(service_pb2_grpc.InferServiceServicer):
    
    # Implement the StartInfer method
    async def StartInfer(self, request, context):
        try:
            # Call the infer function (assumed to be asynchronous)
            response_data = await infer(request.image_path)

            # Return the gRPC response with necessary fields
            return service_pb2.InferResponse(
                response=json.dumps(response_data),  # Convert the dictionary to a JSON string
                logId=request.logId,
                sessionId=request.sessionId
            )
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            context.set_details(f'Error during inference: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.InferResponse()

async def serve():
    # Create a gRPC server
    server = grpc.aio.server()
    
    # Add the service to the server
    service_pb2_grpc.add_InferServiceServicer_to_server(InferServiceServicer(), server)
    
    # Bind the server to a port
    server.add_insecure_port('[::]:7777')
    await server.start()
    print("Server started on port 7777")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)

if __name__ == '__main__':
    asyncio.run(serve())
