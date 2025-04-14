import os
import sys
import argparse
import random
import re
import time
import numpy as np
import torch
import warnings
import streamlit as st
from transformers import AutoTokenizer
from model import MiniMindLM
from LMConfig import LMConfig
from pathlib import Path

# === 页面配置与资源 ===
st.set_page_config(page_title="WhoSay8dao", layout="centered", initial_sidebar_state="collapsed")

# 使用 Space 本地 logo 会更快，假设你把 logo 命名为 logo.png
logo_default = "logo.png"
logo_angry = "logo_angry.png"

# 选择头像逻辑（根据情绪关键词）
def choose_avatar(content):
    content = content.lower()
    if any(word in content for word in ["不懂", "烦", "哼", "无语", "气死", "抱歉", "自我"]):
        return logo_angry
    return logo_default

col0, col1, col2 = st.columns([3, 1, 7])
with col1:
    st.image("logo.png")
with col2:
    st.markdown("""
    <div style='padding-top:4px; font-size: 32px; font-weight: bold;'>WhoSay8dao</div>
    """, unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center;color:gray;font-style:italic;margin-top:0">
    谁在胡说八道？当然是我啦～
</p>
<hr style="margin-top:10px;margin-bottom:10px">
""", unsafe_allow_html=True)


# === 模型参数设置栏 ===
st.sidebar.title("模型设定调整")
st.sidebar.text("【注】训练数据偏差，增加上下文记忆时\n多轮对话（较单轮）容易出现能力衰减")
history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
top_p = st.sidebar.slider("Top-P", 0.8, 0.99, 0.85, step=0.01)
temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

# === 模型加载与推理相关 ===
@st.cache_resource
def load_model_tokenizer():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=temperature, type=float)
    parser.add_argument('--top_p', default=top_p, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--n_layers', default=4, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--history_cnt', default=history_chat_num, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--load', default=0, type=int)
    parser.add_argument('--model_mode', default=1, type=int)
    args = parser.parse_args([])

    tokenizer = AutoTokenizer.from_pretrained('.')
    model = MiniMindLM(LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    ))
    ckp = './full_sft_128.pth' if os.path.exists('./full_sft_128.pth') else './pytorch_model.bin'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

    return model.eval().to(args.device), tokenizer

model, tokenizer = load_model_tokenizer()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === 聊天历史状态 ===
if "history" not in st.session_state:
    st.session_state.history = []

# === 显示历史对话 ===
for role, content in st.session_state.history:
    avatar = choose_avatar(content) if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(content, unsafe_allow_html=True)

# === 处理用户输入 ===
prompt = st.chat_input("来，让我胡说八道一下～")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.history.append(("user", prompt))

    with st.chat_message("assistant", avatar=logo_default):
        placeholder = st.empty()
        history = st.session_state.history[-(history_chat_num * 2):] if history_chat_num else [("user", prompt)]
        chat_messages = [{"role": role, "content": content} for role, content in history]
        new_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)[-8191:]

        x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)
        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p, stream=True)
            try:
                for y in res_y:
                    answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                    if (answer and answer[-1] == ' ') or not answer:
                        continue
                    placeholder.markdown(answer, unsafe_allow_html=True)
            except StopIteration:
                answer = "...好像哪里出错了？"

            final_response = answer.replace(new_prompt, "")
            avatar = choose_avatar(final_response)
            st.session_state.history.append(("assistant", final_response))
