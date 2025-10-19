# Neural Machine Translation: Sequence-to-Sequence with Attention for English-to-Vietnamese Translation

This repository presents a robust implementation of a neural machine translation (NMT) system for English-to-Vietnamese translation, inspired by *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau et al., 2014). Modifications to the original architecture enhance performance, incorporating bidirectional encoder state concatenation for improved decoder initialization. The implementation leverages **PyTorch** for efficient tensor operations and automatic differentiation, with additional inspiration from the *Attention is All You Need* Transformer paradigm (Vaswani et al., 2017) for potential extensions.

## Overview

Neural machine translation employs end-to-end neural networks to map source-language sequences to target-language equivalents, surpassing traditional statistical methods by capturing long-range dependencies and semantic nuances. This implementation uses a sequence-to-sequence (Seq2Seq) model with additive attention, balancing computational efficiency and translation fidelity in a low-resource setting.

## Installation

To replicate this project, ensure the following dependencies are installed. Use a Python virtual environment for isolation.

### Prerequisites
- **Python**: Version 3.8 or higher
- **PyTorch**: Version 2.0 or higher (GPU support recommended for faster training)
- **NLTK**: For BLEU score computation
- **NumPy**: For numerical operations
- **tqdm**: For progress bars during training
- **torchtext**: For dataset handling and preprocessing

### Installation Steps
1. **Create a Virtual Environment**:
   ```bash
   python -m venv nmt_env
   source nmt_env/bin/activate  # On Windows: nmt_env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   pip install nltk numpy tqdm torchtext==0.15.2
   ```

3. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('bleu_score')
   ```

4. **Dataset Acquisition**:
   Download the IWSLT'15 English-Vietnamese dataset (small variant) from the Stanford NLP Group:
   ```bash
   wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
   wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi
   wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en
   wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi
   wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
   wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi
   ```

5. **Hardware Recommendations**:
   - GPU: NVIDIA CUDA-compatible (e.g., GTX 1080 or higher) for accelerated training
   - RAM: Minimum 16GB for efficient batch processing
   - Storage: At least 10GB for dataset and model checkpoints

## Dataset

The IWSLT'15 English-Vietnamese dataset comprises:
- **Training**: 133,317 parallel sentence pairs (`train.en`, `train.vi`)
- **Validation**: 1,553 pairs (`tst2012.en`, `tst2012.vi`)
- **Test**: 1,268 pairs (`tst2013.en`, `tst2013.vi`)

### Preprocessing
To ensure computational efficiency, sentences longer than 20 tokens are filtered, retaining ~80% of the corpus. Preprocessing steps include:
1. Tokenization using NLTK's `word_tokenize`.
2. Removal of punctuation and digits using regex (`re.sub(r'[^\w\s]', '', text)`).
3. Lowercasing all tokens.
4. Addition of `<START>` and `<END>` tokens.
5. Padding sequences to a uniform length of 20 tokens with `<PAD>`.

The vocabulary is built dynamically, mapping tokens to indices, with special tokens `<UNK>`, `<PAD>`, `<START>`, and `<END>`. Word embeddings are initialized randomly or optionally pretrained (e.g., GloVe for English).

## Sequence-to-Sequence Architecture

The Seq2Seq model employs an encoder-decoder framework with attention. Given a source sequence \(\mathbf{x} = (x_1, \dots, x_T)\) and target sequence \(\mathbf{y} = (y_1, \dots, y_S)\), the model optimizes:

\[
p(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{S} p(y_t \mid y_{1:t-1}, \mathbf{x}).
\]

### Encoder
A bidirectional Gated Recurrent Unit (GRU) processes the source sequence. For token \(x_i\), the hidden state is:

\[
\mathbf{h}_i = [\overrightarrow{\mathbf{h}}_i; \overleftarrow{\mathbf{h}}_i],
\]

where \(\overrightarrow{\mathbf{h}}_i\) and \(\overleftarrow{\mathbf{h}}_i\) are forward and backward GRU outputs, respectively. The GRU update is defined as:

\[
\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z),
\]
\[
\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r),
\]
\[
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h),
\]
\[
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
\]

The initial decoder state \(\mathbf{s}_0\) combines the final forward and backward states:

\[
\mathbf{s}_0 = \tanh(\mathbf{W}_s [\overrightarrow{\mathbf{h}}_T; \overleftarrow{\mathbf{h}}_1] + \mathbf{b}_s).
\]

This modification enhances cross-lingual alignment compared to using only \(\overrightarrow{\mathbf{h}}_T\).

### Decoder
A unidirectional GRU generates tokens autoregressively. At timestep \(t\), the input is:

\[
\mathbf{s}_t = \text{GRU}([\mathbf{e}(y_{t-1}); \mathbf{c}_t], \mathbf{s}_{t-1}),
\]

where \(\mathbf{e}(y_{t-1})\) is the embedding of the previous token, and \(\mathbf{c}_t\) is the attention-derived context vector. The output probability is:

\[
p(y_t) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t + \mathbf{b}_o).
\]

### Attention Mechanism
The additive attention mechanism (Bahdanau et al., 2014) computes alignment scores:

\[
e_{t,i} = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i),
\]

with normalized weights:

\[
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}.
\]

The context vector is:

\[
\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i} \mathbf{h}_i.
\]

This contrasts with the scaled dot-product attention in Transformers (Vaswani et al., 2017):

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\]

where \(d_k\) is the key dimension, offering parallelization benefits absent in RNNs.

## Training Protocol

The model minimizes cross-entropy loss:

\[
\mathcal{L} = -\sum_{t=1}^{S} \log p(y_t \mid y_{1:t-1}, \mathbf{x}).
\]

### Hyperparameters
- **Optimizer**: Adam (\(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\epsilon = 10^{-8}\))
- **Learning Rate**: 0.001, decayed by 0.2 per epoch via `lr_scheduler.StepLR`
- **Batch Size**: 64
- **Gradient Clipping**: Norm threshold of 5.0
- **Teacher Forcing Ratio**: 1.0 (full teacher forcing)
- **Hidden Size**: 256 for both encoder and decoder GRUs
- **Embedding Size**: 300
- **Dropout**: 0.1 (applied to GRU outputs)

Training monitors validation BLEU scores, with early stopping (patience of 3 epochs) to prevent overfitting.

## Inference Procedure
Inference employs greedy decoding (argmax sampling) or beam search (width 4). Decoding starts with `<START>` and terminates at `<END>` or a maximum length of 50 tokens.

## Evaluation Metric: BLEU Score
The BLEU score (Papineni et al., 2002) evaluates n-gram precision:

\[
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{4} \frac{1}{4} \log p_n \right),
\]

where \(p_n\) is the modified n-gram precision, and \(\text{BP} = \min(1, e^{1 - \frac{r}{c}})\) penalizes brevity (\(r\): reference length, \(c\): candidate length). NLTK's `sentence_bleu` is used with uniform weights.

### Results
After 5 epochs on Google Colab (no early stopping):

| Model       | BLEU Score (Test Set) |
|-------------|-----------------------|
| Original    | 18.524                |
| Modified    | 19.283                |

The modified model’s superior performance stems from enhanced decoder initialization. Hyperparameter sensitivity necessitates grid search or Bayesian optimization.

## Qualitative Examples
Test set translations after 5 epochs:

1. **Source**: "and i was very proud"  
   **Original**: và tôi rất tự hào  
   **Modified**: và tôi tự hào rất tự hào  

2. **Source**: "but most people don’t agree"  
   **Original**: nhưng hầu hết mọi người không đồng ý  
   **Modified**: nhưng hầu hết mọi người không đồng ý  

3. **Source**: "i also didn’t know that the second step is to isolate the victim"  
   **Original**: tôi cũng không biết rằng thứ hai là để phân loại các nạn nhân  
   **Modified**: tôi cũng không biết rằng bước thứ hai là để chuyển nạn nhân  

4. **Source**: "my family was not poor and myself i had never experienced hunger"  
   **Original**: gia đình tôi không phải là nghèo và tôi không bao giờ hồi phục hồi  
   **Modified**: gia đình tôi không nghèo và tôi không bao giờ có thể nhìn qua đói  

5. **Source**: "this was the first time i heard that people in my country were suffering"  
   **Original**: lần đầu tiên tôi nghe thấy mọi người ở đất nước của tôi bị đau khổ  
   **Modified**: đó là lần đầu tiên tôi nghe thấy mọi người ở đất nước của tôi rất đau khổ  

Repetitions and semantic drifts suggest undertraining, addressable via extended epochs or scheduled sampling.

## Complete Architecture
The model integrates:
- Bidirectional GRU encoder
- Unidirectional GRU decoder
- Additive attention
- Concatenated terminal state initialization

Future extensions could adopt Transformer’s multi-head attention:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O,
\]
\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V).
\]

## Potential Enhancements
1. **Subword Tokenization**: Use SentencePiece or BPE to handle rare words.
2. **Regularization**: Apply dropout (0.3) and label smoothing (\(\epsilon = 0.1\)).
3. **Decoding**: Implement nucleus sampling (\(p = 0.9\)) or diverse beam search.
4. **Pretraining**: Initialize embeddings with multilingual BERT or XLM-R.
5. **Metrics**: Include TER, METEOR, or human evaluations.
6. **Efficiency**: Use mixed-precision training (FP16) via `torch.cuda.amp`.
7. **Analysis**: Visualize attention weights to diagnose misalignments.

## References
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473.
- Vaswani, A., et al. (2017). *Attention is All You Need*. Advances in Neural Information Processing Systems, 30.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.
- Papineni, K., et al. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation*. ACL.
- Bentrevett’s PyTorch Seq2Seq Tutorial: https://github.com/bentrevett/pytorch-seq2seq
- TensorFlow NMT Tutorial: https://www.tensorflow.org/tutorials/text/nmt_with_attention
- TensorFlow NMT Repository: https://github.com/tensorflow/nmt
- Viblo Article: https://viblo.asia/p/machine-translate-voi-attention-trong-deep-learning-Az45bY7zlxY
- Coursera Sequence Models (Week 3) by Andrew Ng

Contributions are welcome; prioritize empirical validation and computational efficiency.
