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
如果是只做bit search，結果應該長這這樣：
<img src="https://github.com/user-attachments/assets/018f084d-c562-458a-8ba6-cafdb969da18" alt="image" width="500">

如果是做finetune，應該長這樣：
![image](https://github.com/user-attachments/assets/4c109b50-75bb-4c09-b836-4ce2f50c7a4e)

<img src="https://github.com/user-attachments/assets/4c109b50-75bb-4c09-b836-4ce2f50c7a4e" alt="image" width="500">
## 4. Implementation (Task - Text Generation)
```bash
cd LLM-MIXED-PRECISION/llm-mixed-precision/text_generation/
bash search.sh
```
以GPT2為例，在search.sh設定bitw, model, 是否要finetune等，如圖：
![image](https://github.com/user-attachments/assets/eb4f2207-a28c-45ed-a8c0-da94621cf269)

