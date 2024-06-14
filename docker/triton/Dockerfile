FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Expose the Triton server port
EXPOSE 8000

# Start the Triton server with the model repository
CMD ["tritonserver", "--model-repository=/models"]