# No. DC-AI-C12: Image compression
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain is to compress the images and reduce the redundancy.

## How to run

### Training

```bash
./train.sh
```

### Encode
`
python encoder.py --model checkpoint/encoder_epoch_00000005.pth --input /path/to/your/example.png --cuda --output ex --iterations 16
`

This will output binary codes saved in `.npz` format.

### Decode
`
python decoder.py --model checkpoint/encoder_epoch_00000005.pth --input /path/to/your/example.npz --cuda --output /path/to/output/folder
`

This will output images of different quality levels.
