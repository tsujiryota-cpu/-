"""学習済み重みを ONNX に書き出す。 usage: python3 export.py det.pt cls.pt OUT_DIR"""
import sys, torch, numpy as np, onnxruntime as ort
from train_det import Det
from train_cls import Cls
det_pt, cls_pt, out = sys.argv[1:4]
d = Det(); d.load_state_dict(torch.load(det_pt)); d.eval()
sd = torch.load(cls_pt)
w = [sd[f'f.{i}.0.weight'].shape[0] for i in range(6)]
c = Cls(w, sd['f.8.weight'].shape[0]); c.load_state_dict(sd); c.eval()
x = torch.rand(1, 3, 352, 640)
torch.onnx.export(d, (x,), out + '/det.onnx', input_names=['image'], output_names=['out'], opset_version=17, dynamo=False,
                  dynamic_axes={'image': {2: 'h', 3: 'w'}, 'out': {2: 'gh', 3: 'gw'}})
y = torch.rand(5, 3, 40, 32)
torch.onnx.export(c, (y,), out + '/cls.onnx', input_names=['crops'], output_names=['prob'], opset_version=17, dynamo=False,
                  dynamic_axes={'crops': {0: 'n'}, 'prob': {0: 'n'}})
for name, net, inp, key in (('det', d, torch.rand(1, 3, 544, 960), 'image'), ('cls', c, torch.rand(9, 3, 40, 32), 'crops')):
    s = ort.InferenceSession(f'{out}/{name}.onnx')
    a = s.run(None, {key: inp.numpy()})[0]
    b = net(inp).detach().numpy()
    print(name, a.shape, 'max diff', float(np.abs(a - b).max()))
