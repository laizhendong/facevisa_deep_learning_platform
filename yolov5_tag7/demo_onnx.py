
import onnxruntime
import numpy as np
import onnx,cv2
import argparse
import torch
from utils.general import non_max_suppression,scale_boxes

ap = argparse.ArgumentParser(description="run onnx")
ap.add_argument("-model",help="model path",default="runs/train/exp4-0501/weights/last.onnx")
ap.add_argument("-size",help="input image size",default=640,type=int)


def decoder_output(output):
    x,anchor_grid,stride, grid = output[0:3], output[3],output[4], output[5:]
    z = []
    for i in range(len(x)):
        y = torch.from_numpy(x[i]).sigmoid().numpy()
        bs,na,ny,nx,no = y.shape
        #nc = na - 5
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        #z.append(y.view(bs, -1, no))
        z.append(np.reshape(y,(bs,-1,no)))
    #return torch.cat(z,1).numpy()
    return np.concatenate(z,axis=1)

def run_onnx(model_path, input_shape):
    # random input tensor 
    input_tensor = np.random.randn(*input_shape).astype(np.float32)
    image_raw = cv2.imread("data/images/defect_101745_38_dty_f31111af1_20230130103643_original_ww_q100_a@1675046203770_7_80_1806_437.jpg",1)
    image_data = cv2.resize(image_raw,(input_shape[-1],input_shape[-2]),interpolation=cv2.INTER_LINEAR)
    image_data = image_data.astype(np.float32) / 255.0
    image_data = np.transpose(image_data[:,:,::-1],(2,0,1))
    input_tensor[0] = image_data

    ort_session = onnxruntime.InferenceSession(model_path)


    # compute onnx Runtime output prediction
    onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    pred = ort_session.run(None, onnx_inputs)
    pred = non_max_suppression([torch.from_numpy(p) for p in pred],0.1,0.5)
    for i, det in enumerate(pred):  # detections per image
        if det is None or len(det) < 1:
            continue
        im0 = image_data
        im0sz = image_raw.shape
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im0.shape[1:], det[:, :4], im0sz).round()

        for i_box in det:
            x0,y0,x1,y1 = [int(i_box[k]) for k in range(4)]
            cv2.rectangle(image_raw, (x0, y0), (x1,y1), (0, 255, 255), 1, 1)
            cv2.putText(image_raw, "p" + str(i_box[4].item()), (x0+30, y0+30), 2, 1.5,  (0, 255, 255))

        # Stream results
        cv2.imshow("output", image_raw)
        cv2.imwrite("output.jpg",image_raw)
        if cv2.waitKey(-1) == ord('q'):  # q to quit
            raise StopIteration

def main(args):
    run_onnx(args.model,(1,3,args.size,args.size))
if __name__ == "__main__":
    args = ap.parse_args()
    main(args)