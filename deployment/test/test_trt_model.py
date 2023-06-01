import numpy as np
from pathlib import Path
from src.ai.trt_model import TrtModel


# run with python3 -m pytest from tensorrt directory
def test_prediction():
    onnx_file = Path("onnx/fairmot_dla34.onnx")
    assert onnx_file.is_file()
    predictor = TrtModel(onnx_file)
    x_file = Path("test/model", onnx_file.name.replace(".onnx", "_x.npy"))
    x = np.load(x_file)
    yp_s = predictor(x)
    # assert list(yp_s.keys()) == ["wh", "reg", "id", "hm"]
    for k, yp in yp_s.items():
        yi_file = Path("test/model", onnx_file.name.replace(".onnx", f"_y_{k}.npy"))
        yi = np.load(yi_file)
        np.testing.assert_allclose(yp, yi, atol=1e-2, err_msg=f"error in {k}")
