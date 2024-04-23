## 下载模型的命令
```
export HF_ENDPOINT=https://hf-mirror.com/
# 下载模型
huggingface-cli download --resume-download codellama/CodeLlama-7b-hf
huggingface-cli download --resume-download meta-llama/Meta-Llama-3-8B

# 下载数据集

```

* 多卡机器，如何指定其中部分卡
通过环境变量，指定2、4、5、7号卡

```bash
export CUDA_VISIBLE_DEVICES=2,4,5,7
```

* bitsandbytes[cuda]使用cuda版本    
量化使用的库

```bash
pip install "bitsandbytes[cuda]"
```
