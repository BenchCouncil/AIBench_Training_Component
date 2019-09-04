# AIBench Component Benchmark

AIBench: An Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

The user manual of AIBench:
http://www.benchcouncil.org/AIBench/files/AIBench-User-Manual.pdf

The component benchmarks of AIBench includes 16 workloads, covering 16 prominent
AI problem domains, including classification, image generation, text-to-text 
translation, image-to-text, image-to- image, speech-to-text, face embedding, 
3D face recognition, object detection, video prediction, image compression, 
recommendation, 3D object reconstruction, text summarization, spatial 
transformer, and learning to rank.  

The workloads are implemented using the state-of-the-art neural network models
and the state-of-the-practice deep learning frameworks including TensorFlow and 
PyTorch.

**No. DC-AI-C1: Image classification**. This benchmark is to extract different thematic classes 
within an image, which is a supervised learning problem to define a set of tar
get classes and train a model to recognize. This benchmark uses ResNet neural 
network and uses ImageNet as data input.

**No. DC-AI-C2: Image generation**. This benchmark provides an unsupervised learning problem to 
mimic the distribution of data. It uses WGAN algorithms and uses LSUN dataset 
as data input.

**No. DC-AI-C3: Text-to-Text Translation**. This benchmark translates text from one language to 
another, which is the most important field of computational linguistics. It 
adopts recurrent neural networks with an encoder and a decoder, and takes 
WMT English-German as data input.

**No. DC-AI-C4: Image-to-Text**. This benchmark generates the description of an image automatically. 
It uses Neural Image Caption consisting of a vision CNN followed by a language 
generating RNN model and takes Microsoft COCO dataset as input.

**No. DC-AI-C5: Image-to-Image**. This benchmark converts an image from one representation of a 
specific scene to another scene or representation. It uses the cycle-GAN 
algorithm and takes Cityscapes dataset as input.

**No. DC-AI-C6: Speech recognition**. This benchmark recognizes and translates the spoken language to 
text. It uses the DeepSpeech2 algorithm and takes Librispeech dataset as input.

**No. DC-AI-C7: Face embedding**. This benchmark transforms a facial image to a vector in embedding 
space. It uses the FaceNet algorithm and takes the LFW (Labeled Faces in the 
Wild) dataset or VGGFace2 as input.

**No. DC-AI-C8: 3D face recognition**. This benchmark recognize the 3D facial information from an 
image. It provides a series of 3D face modes. The test data is provided by 
Intellifusion and includes 77,715 samples from 253 face IDs, which are published 
on the BenchCouncil web site.

**No. DC-AI-C9: Object detection**. This benchmark detects the objects within an image. It uses 
the Faster R-CNN algorithm and takes Microsoft COCO dataset as input.

**No. DC-AI-C10: Recommendation**. This benchmark provides movie recommentations. It uses 
collaborative filtering algorithm and takes MovieLens dataset as input.

**No. DC-AI-C11: Video prediction**. This benchmark predicts the future video through predicting 
previous frames transformation. It uses motion-focused predictive models and 
takes Robot pushing dataset as input.

**No. DC-AI-C12: Image compression**. This benchmark provides full-resolution lossy image
compression. It uses recurrent neural networks and takes ImageNet dataset as input.

**No. DC-AI-C13: 3D object reconstruction**. This benchmark predicts and reconstructs 3D objects. 
It uses a convolutional encoder-decoder network and takes ShapeNet Dataset as input.

**No. DC-AI-C14: Text summarization**. This benchmark is to generate the text summary. It uses 
sequence-to-sequence model and takes Gigaword dataset as input.

**No. DC-AI-C15: Spatial transformer**. This benchmark performs spatial transformations. It uses 
spatial transformer networks and takes MNIST dataset as input.

**No. DC-AI-C16: Learning to rank**. This benchmark learns the attributes of searched content 
and rank the scores for the results, which is the key for searching service. It 
uses ranking distillation model and takes gowalla dataset as input.

