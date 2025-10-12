import os
import sys

# Fix Triton/CUDA compilation issues before importing any CUDA libraries
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', '/usr/local/cuda')
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/lib64:' + os.environ.get('LIBRARY_PATH', '')

# Add linker flags for Triton compilation
os.environ['LDFLAGS'] = '-L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu ' + os.environ.get('LDFLAGS', '')

from llava.train.train import train

if __name__ == "__main__":
    train()
