import torch
import tensorflow as tf

# torch
#print(f"torch : MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
#print(f"torch : MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")

#device = torch.device("mps")

# tensorflow
print("TensorFlow has access to the following devices:", tf.config.list_physical_devices())
print("TensorFlow version : ", tf.__version__)