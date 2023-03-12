## RVQ-VAE-GPT - Residual Vector Quantize VAE - GPT (wip)

My attempts at applying <a href="https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py">Soundstream</a> design on learned tokenization of text and then applying a <a href="https://github.com/lucidrains/RQ-Transformer/blob/main/rq_transformer/hierarchical_causal_transformer.py">hierarchical transformer</a> to text generation.

The Soundstream will be modified to use all local attention. Experiments will compare VQ, RVQ, and also multi-headed VQ

Was told by a researcher friend this will likely fail 😂😂 but I will try it anyways, yolo. In the case it does not work, maybe it can still be useful for genomics. Come to think of it, why shouldn't it be able to at least learn bigrams (for english) and codons (for genomics)?

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2107.03312,
  title  = {SoundStream: An End-to-End Neural Audio Codec},
  author = {Zeghidour, Neil and Luebs, Alejandro and Omran, Ahmed and Skoglund, Jan and Tagliasacchi, Marco},
  publisher = {arXiv},
  url    = {https://arxiv.org/abs/2107.03312},
  year   = {2021}
}
```

```bibtex
@unknown{unknown,
    author  = {Lee, Doyup and Kim, Chiheon and Kim, Saehoon and Cho, Minsu and Han, Wook-Shin},
    year    = {2022},
    month   = {03},
    title   = {Autoregressive Image Generation using Residual Quantization}
}
```

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```
