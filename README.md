# TeraQuant: Text-Related Adversarial Quantization for Mixed Precision Large Language Models

TeraQuant is a framework designed to optimize mixed precision quantization for large language models (LLMs). It minimizes distortions in text-related tasks while improving memory efficiency and computational performance through adversary-aware bit allocation.

---

## 1. Installation 
Create conda env
```bash
conda env create -f env.yaml
```
```bash
conda activate llm
```
## 2. Preparation
### Datasets
程式中使用了 Hugging Face 的 datasets, 包含ptb, wikitext 以及 c4，可使用以下連結下載，放在 llm-mixed-precision/text_generation/dataset   
https://huggingface.co/datasets/mindchain/wikitext2   
https://huggingface.co/datasets/longisland3/ptb-xl   
https://huggingface.co/datasets/allenai/c4

### Calculation of flops
結果已經在flops_calculation 資料夾中了，不須重跑
```bash
cd LLM-MIXED-PRECISION/llm-mixed-precision/flops_calculation
python cal_flops.py
```
## 3. Outputs

## 4. Implementation
```bash
cd LLM-MIXED-PRECISION/llm-mixed-precision/text_generation/
```
```bash
bash search.sh
```
