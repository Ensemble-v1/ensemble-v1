import torch
import ultralytics

# Add the required classes to the safe globals
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.AIFI,
    ultralytics.nn.modules.C1,
    ultralytics.nn.modules.C2,
    ultralytics.nn.modules.C3,
    ultralytics.nn.modules.C3TR,
    ultralytics.nn.modules.ELAN1,
    ultralytics.nn.modules.OBB,
    ultralytics.nn.modules.PSA,
    ultralytics.nn.modules.SPP,
    ultralytics.nn.modules.SPPELAN,
    ultralytics.nn.modules.SPPF,
    ultralytics.nn.modules.AConv,
    ultralytics.nn.modules.ADown,
    ultralytics.nn.modules.Bottleneck,
    ultralytics.nn.modules.BottleneckCSP,
    ultralytics.nn.modules.C2f,
    ultralytics.nn.modules.C2fAttn,
    ultralytics.nn.modules.C2fCIB,
    ultralytics.nn.modules.C3Ghost,
    ultralytics.nn.modules.C3x,
    ultralytics.nn.modules.CBFuse,
    ultralytics.nn.modules.CBLinear,
    ultralytics.nn.modules.Classify,
    ultralytics.nn.modules.Concat,
    ultralytics.nn.modules.Conv,
    ultralytics.nn.modules.Conv2,
    ultralytics.nn.modules.ConvTranspose,
    ultralytics.nn.modules.Detect,
    ultralytics.nn.modules.DWConv,
    ultralytics.nn.modules.DWConvTranspose2d,
    ultralytics.nn.modules.Focus,
    ultralytics.nn.modules.GhostBottleneck,
    ultralytics.nn.modules.GhostConv,
    ultralytics.nn.modules.HGBlock,
    ultralytics.nn.modules.HGStem,
    ultralytics.nn.modules.ImagePoolingAttn,
    ultralytics.nn.modules.Pose,
    ultralytics.nn.modules.RepC3,
    ultralytics.nn.modules.RepConv,
    ultralytics.nn.modules.RepNCSPELAN4,
    ultralytics.nn.modules.RepVGGDW,
    ultralytics.nn.modules.ResNetLayer,
    ultralytics.nn.modules.RTDETRDecoder,
    ultralytics.nn.modules.SCDown,
    ultralytics.nn.modules.Segment,
    ultralytics.nn.modules.WorldDetect,
    ultralytics.nn.modules.v10Detect
])

if __name__ == '__main__':
    # Load the model
    model = ultralytics.YOLO('yolov8m.pt')

    # Train the model
    model.train(
        data='music_symbols.yaml',
        epochs=5,
        batch=4,
        device='cpu',
        project='runs'
    )
