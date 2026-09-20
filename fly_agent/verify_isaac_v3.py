from isaac_v3 import config
from isaac_v3.retina import FlyRetina, RetinaOutput
from isaac_v3.decoder import ActionDecoder
from isaac_v3.trace import DescendingTrace
from isaac_v3.controller import HybridIsaacController

print("FlyIsaac V3 imports: OK")
print("RETINA:", config.RETINA_ROWS, "x", config.RETINA_COLS)
print("DN_TRACE_DECAY:", config.DN_TRACE_DECAY)
print("DN_TRACE_CLIP:", config.DN_TRACE_CLIP)
print("FlyRetina:", FlyRetina.__name__)
print("RetinaOutput:", RetinaOutput.__name__)
print("ActionDecoder:", ActionDecoder.__name__)
print("DescendingTrace:", DescendingTrace.__name__)
print("HybridIsaacController:", HybridIsaacController.__name__)
