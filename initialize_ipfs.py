import os
import time

ready = os.popen('./go-ipfs/ipfs stats bw').read()
while not ready:
  time.sleep(0.1)
  ready = os.popen('./go-ipfs/ipfs stats bw').read()

print('ipfs is ready')