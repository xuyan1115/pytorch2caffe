## PyTorch2Caffe

Ported from [pytorch2caffe](https://github.com/longcw/pytorch2caffe). 

Add support for 
+ Slice Layer (IndexBackward in PyTorch)

```
We can obtain almost the same output from caffe except Upsampling
for [beyond-part-models](https://github.com/huanghoujing/beyond-part-models): 
diff between pytorch and caffe: min: 0.0, max: 0.466215610504, mean: 0.0415956825018, median: 0.000256508588791
see more in demo.py
```

# Note

This tool only support for PyTorch version less than 0.3.0
I cannot find how to access layer parameters in 0.3.0+ PyTorch :)

