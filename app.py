from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
from dotenv import load_dotenv

# Hugging Face 模型
from transformers import pipeline
import torch
from huggingface_hub import login

# 讀取 .env 環境變數
load_dotenv()
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# 登入 Hugging Face（不會被 GitHub 掃到 token）
login(os.getenv("HF_TOKEN"))

# 初始化 NER 模型
device = 0 if torch.cuda.is_available() else -1
ner = pipeline("ner", model="ckiplab/bert-base-chinese-ner", aggregation_strategy="simple", device=device)

# 讀取氣象 JSON
import json

with open("F-D0047-091.json", encoding="utf-8") as f:
    weather_data = json.load(f)

# 天氣查詢函式
def run_weather_pipeline(user_input):
    entities = ner(user_input)
    locations = [e["word"] for e in entities if e["entity_group"] == "LOC"]
    
    if not locations:
        return "請輸入你想查詢天氣的地點，例如：台北今天會下雨嗎？"

    location = locations[0]  # 預設只取第一個地點
    try:
        location_data = next(
            loc for loc in weather_data["cwbopendata"]["dataset"]["location"]
            if location in loc["locationName"]
        )
    except StopIteration:
        return f"找不到「{location}」的天氣資訊"

    wx_element = next(
        ele for ele in location_data["weatherElement"]
        if ele["elementName"] == "Wx"
    )

    time_block = wx_element["time"][0]
    weather_desc = time_block["elementValue"][0]["value"]
    start = time_block["startTime"]
    end = time_block["endTime"]

    return f"{location} {start} ~ {end} 的天氣為：{weather_desc}"

# === Flask 主程式 ===
app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text
    weather_reply = run_weather_pipeline(user_input)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=weather_reply)
    )

if __name__ == "__main__":
    app.run()




