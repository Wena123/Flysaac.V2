from isaac_v3 import config
from isaac_v3.trace import DescendingTrace
from isaac_v3.retina import FlyRetina

print("isaac_v3 import: OK")
print("DN_TRACE_DECAY:", config.DN_TRACE_DECAY)
print("DN_TRACE_CLIP:", config.DN_TRACE_CLIP)
print("FlyRetina:", FlyRetina.__name__)
print("DescendingTrace:", DescendingTrace.__name__)
