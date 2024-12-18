# LoRAs

本项目展示了如何在模型中使用多个 LoRA（低秩适应）模块，以实现参数高效的微调和模型适应。

## 目录

- [简介](#简介)
- [安装](#安装)
- [使用方法](#使用方法)
  - [加载多个 LoRA 模型](#加载多个-lora-模型)
  - [训练新的 LoRA 模型](#训练新的-lora-模型)
- [示例](#示例)
- [许可证](#许可证)

## 简介

LoRA（低秩适应）是一种参数高效的微调方法，通过在预训练模型的权重上添加低秩矩阵，实现对新任务的适应，而无需对整个模型进行全面微调。

本项目提供了以下内容：

- 如何加载和使用多个 LoRA 模型。
- 如何训练新的 LoRA 模型。
- 示例代码和教程。

此外，我们使用了 [Hugging Face's `datasets` 库](https://github.com/huggingface/datasets) 来处理和加载数据集，使训练和推理更加高效。

## 安装

请确保您的环境中安装了以下依赖项：

- Python 3.7 或更高版本
- PyTorch 1.8 或更高版本
- Transformers 库
- PEFT（Parameter-Efficient Fine-Tuning）库
- Datasets 库

您可以使用以下命令安装所需的 Python 包：

```bash
pip install torch transformers peft datasets
```

## 使用方法

### 加载多个 LoRA 模型

要在模型中加载多个 LoRA 模型，请参考 `PeftModel_LoRAs.ipynb` 和 `PeftMixedModel_LoRAs.ipynb`。

以下是加载多个 LoRA 模型的基本步骤：

1. **加载预训练模型：**

   ```python
   from transformers import AutoModelForSequenceClassification

   base_model = AutoModelForSequenceClassification.from_pretrained('模型名称')
   ```

2. **应用多个 LoRA 模型：**

   ```python
   from peft import PeftModel

   lora_model_1 = PeftModel.from_pretrained(base_model, 'lora_model_1_path')
   lora_model_2 = PeftModel.from_pretrained(base_model, 'lora_model_2_path')

   # 将多个 LoRA 模型的权重合并到基础模型中
   base_model.load_state_dict(lora_model_1.state_dict(), strict=False)
   base_model.load_state_dict(lora_model_2.state_dict(), strict=False)
   ```

### 训练新的 LoRA 模型

要训练新的 LoRA 模型，请参考 `PeftMixedModel_LoRAs_Training.ipynb`。

以下是训练 LoRA 模型的基本步骤：

1. **加载数据集：**

   使用 `datasets` 库加载您的数据集：

   ```python
   from datasets import load_dataset

   dataset = load_dataset('imdb')  # 示例数据集
   train_data = dataset['train']
   test_data = dataset['test']
   ```

2. **定义 LoRA 配置：**

   ```python
   from peft import LoraConfig

   lora_config = LoraConfig(
       r=8,
       lora_alpha=32,
       target_modules=['q_proj', 'v_proj'],
       lora_dropout=0.1,
       bias='none'
   )
   ```

3. **创建 LoRA 模型：**

   ```python
   from peft import get_peft_model

   lora_model = get_peft_model(base_model, lora_config)
   ```

4. **训练 LoRA 模型：**

   使用标准的 PyTorch 训练循环，结合 `datasets` 数据加载器，训练 `lora_model`，并保存训练后的模型。

## 示例

有关如何使用和训练 LoRA 模型的详细示例，请参阅以下 Jupyter Notebook：

- `PeftModel_LoRAs.ipynb`
- `PeftMixedModel_LoRAs.ipynb`
- `PeftMixedModel_LoRAs_Training.ipynb`

## 许可证

本项目采用 GPL-3.0 许可证。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
