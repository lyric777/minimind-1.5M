### 真正的穷鬼版，只用colab免费GPU
### 所有的介绍、教程见伟大的https://github.com/jingyaogong/minimind

#### 1. 修改了requirements文件，目前colab 用的是python3.11解释器，python版本和各种包版本冲突真的很烦（该配置不是唯一选择）


#### 2. 仅为了体验训练流程，调整了预训练参数，最终达到可以用免费GPU运行时长训练的结果，总共不到50分钟（sft用了不到40分钟）
train_pretrain.py 和 train_full_sft.py main中
```bash
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--n_layers', default=4, type=int)
```

为什么改这两个不改其他，参考了该issue https://github.com/jingyaogong/minimind/issues/35
##### 用原模型（最小版）参数训练，预训练就预计花费两个半小时，而colab T4 GPU的限制大概是1个小时40分钟（在当时的用量水平下），我已经踩过坑了，不用再尝试了
同时，如果在colab上执行，需要注意路径问题，我在代码里都使用了挂载Google drive后的绝对路径

#### 3. 数据集Google drive分享
pretrain_hq.jsonl：

https://drive.google.com/file/d/11lURXJgRllu3wjFpaVXkgh1fPLFrUyz4/view?usp=sharing

sft_mini_512.jsonl:

https://drive.google.com/file/d/11qdi7EKocdixTcGr692RIVmY3I6pK_Sw/view?usp=sharing

右边菜单栏选择“将快捷方式添加到云端硬盘”

#### 4. 最后恭喜你啊，训练出智障了
```bash
MiniMind模型参数量: 1.57M(illion)
[0] 自动测试
[1] 手动输入
0
👶: 请介绍一下自己。
🤖️: 在我做什么吗？

👶: 你更擅长哪一个学科？
🤖️: 这个学科特的学科科是一种生物学科的学科。
学科的学科，教学科： 学科。学科科科科科科科科，哲学科。学科兵：读进学科科科学，学科中。

👶: 鲁迅的《狂人日记》是如何批判封建礼教的？
🤖️: 鲸语上的女孩通过人物来挣扎，突出对方的利益和喜爱和感情，最终证明了他的性格和个人。

👶: 我咳嗽已经持续了两周，需要去医院检查吗？
🤖️: 我想到了最高的桌面，请问你在哪个地方做哪个地方？
```
