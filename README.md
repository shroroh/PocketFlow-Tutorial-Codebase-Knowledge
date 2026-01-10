<h1 align="center">Teacher AI</h1>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
 <a href="https://discord.gg/hUE6T">
    <img src="https://img.shields.io/discord/1346833819172601907?logo=discord&style=flat">


## 🚀 Getting Started

1. Clone this repository
   ```bash
  git clone https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge](https://github.com/shroroh/TeacherFlow-.git
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up LLM in [`utils/call_llm.py`](./utils/call_llm.py) by providing credentials. To do so, you can put the values in a `.env` file. By default, you can use the AI Studio key with this client for Gemini Pro 2.5 by setting the `GEMINI_API_KEY` environment variable. If you want to use another LLM, you can set the `LLM_PROVIDER` environment variable (e.g. `XAI`), and then set the model, url, and API key (e.g. `XAI_MODEL`, `XAI_URL`,`XAI_API_KEY`). If using Ollama, the url is `http://localhost:11434/` and the API key can be omitted.
   You can use your own models. We highly recommend the latest models with thinking capabilities (Claude 3.7 with thinking, O1). You can verify that it is correctly set up by running:
   ```bash
   python utils/call_llm.py

   ИЛИ в среде исполнения, например консоль Power Shell заводим переменные окружения перед запуском основным.
   
   $env:GEMINI_API_KEY="AIzaSyBXEcK_E..|\/..GsWrG"
   $env:GEMINI_MODEL="gemini-flash-latest"
   ```

5. Excute main.py
 ```bash
 python main.py --student-id ivan123 --no-cache
 ```


<details>
 

<br>
<div align="center">
  <a href="https://youtu.be/AFY67zOpbSo" target="_blank">
    <img src="./assets/youtube_thumbnail.png" width="500" alt="Pocket Flow Codebase Tutorial" style="cursor: pointer;">
  </a>
</div>
<br>



