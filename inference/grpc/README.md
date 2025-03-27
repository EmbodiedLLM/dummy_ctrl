# gRPC Protocol Documentation

## Compiling Protocol Buffers

When you make changes to the file transfer format in the `.proto` files, you need to recompile the protocol buffers using the following command:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. policy.proto
```
