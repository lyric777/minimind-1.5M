## 分词器训练（train_tokenizer）
（1）数据读取
pretrain_hq.jsonl 文件中存储了预训练的文本数据，每一行是一个 JSON 对象，text 字段包含具体文本内容。

代码通过 read_texts_from_jsonl 迭代读取 text 字段的内容作为训练语料。

（2）分词器初始化
你使用了 BPE（Byte Pair Encoding） 作为模型：
tokenizer = Tokenizer(models.BPE())

预分词器采用 ByteLevel（字节级拆分）：
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
ByteLevel 预分词器能够处理 字节级别的文本，常用于 BPE 和 WordPiece 分词，适用于各种语言，包括中文。

### 为什么用ByteLevel   为了中文utf-8
特性	标准 BPE	ByteLevel BPE
输入处理	Unicode 字符	UTF-8 字节
多语言支持	一般（依赖词汇表）	极强（字节级泛化能力）
稀有字符处理	可能失败	总能分解为字节
典型应用	早期 GPT-2	GPT-3/4、RoBERTa、XLM-R

（3）分词器训练
你设定了 BpeTrainer 训练参数：
trainer = trainers.BpeTrainer(
    vocab_size=6400,  # 词汇表大小 6400
    special_tokens=special_tokens,  # 包含特殊 token
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
词汇表大小：6400（相对较小，适合轻量级 LLM）。
特殊 token：
<unk>（未知 token，对应索引 0）。
<s>（句子开头，对应索引 1）。
</s>（句子结束，对应索引 2）。
初始字母表：ByteLevel.alphabet() 允许模型直接处理 UTF-8 编码的字符，而无需额外的字符级别转换。


训练方式：
tokenizer.train_from_iterator(texts, trainer=trainer)
直接从 JSONL 文件中迭代文本进行训练。

（4）设置解码器
你为 tokenizer 设置了 ByteLevel 解码器，确保能正确解码成原始文本：
tokenizer.decoder = decoders.ByteLevel()

（5）存储分词器
你将训练好的分词器 保存到 ../model/minimind_tokenizer/ 目录：
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save("../model/minimind_tokenizer")

（6）手动创建配置文件
你定义了 tokenizer_config.json，其中包括：
最大长度：32768（用于限制模型输入）。
特殊 token 配置。
chat_template 模板（适用于多轮对话）。
默认 pad_token 设置为 <unk>（因为没有额外的 <pad> token）。
