import numpy as np
import cv2

from isaac_v3.retina import FlyRetina


def frame_at(x):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (x, 95), (x + 28, 123), (255, 255, 255), -1)
    return img


def main():
    retina = FlyRetina()
    retina.process(frame_at(70))
    out = retina.process(frame_at(105))

    assert out.motion.max() > 0.05, "motion detector is silent"
    assert out.target.max() > 0.02, "target salience is silent"
    assert out.motion_right.max() > out.motion_left.max(), "rightward motion not detected"
    print("RETINA TEST: OK")
    print("motion max      :", float(out.motion.max()))
    print("target max      :", float(out.target.max()))
    print("right/left max  :", float(out.motion_right.max()), "/", float(out.motion_left.max()))


if __name__ == "__main__":
    main()
