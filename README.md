# LangChain Course — Hello World Setup

Steps to follow along with the first video.

## 1. Fork the Repository

Go to [emarco177/langchain-course](https://github.com/emarco177/langchain-course) and click **Fork**.

## 2. Create a New Branch

```bash
git checkout --orphan project/hello-world
git rm -rf .
```

## 3. Initialize the Project

```bash
uv init
uv add langchain langchain-openai langchain-ollama python-dotenv black isort
```

## 4. Configure Environment Variables

Create a `.env` file at the root of the project:

```env
OPENAI_API_KEY=your_key_here

# LangSmith tracing (optional but recommended)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT="your_project_name"
```

- Get your OpenAI API key at [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- Get your LangSmith API key at [smith.langchain.com/settings](https://smith.langchain.com/settings)
- You can use any other LLM provider instead of OpenAI

## 5. Push Your Branch

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/langchain-course.git
git push --set-upstream origin project/hello-world
```

Replace `YOUR_USERNAME` with your GitHub username.

## Using an Open-Source LLM with Ollama

1. Download Ollama from [ollama.com/download](https://ollama.com/download)
2. Browse models at [ollama.com/search](https://ollama.com/search) and pull one:
   ```bash
   ollama pull gemma3:4b
   ```
3. List downloaded models:
   ```bash
   ollama list
   ```
4. Run a model in the terminal:
   ```bash
   ollama run gemma3:4b
   ```
5. Use it in Python — the package is already included in step 3:
   ```python
   from langchain_ollama import ChatOllama
   llm = ChatOllama(model="gemma3:4b", temperature=0.0)
   ```

## Trace the LLM calls with LangSmith

1. register langsmith account, and get the API key from https://smith.langchain.com/account/api-keys, then set it to the LANGSMITH_API_KEY environment variable. You can also set the LANGCHAIN_PROJECT environment variable to your project name, which will help you organize your traces in LangSmith.
2. setup the .env file in the root directory of your project follwoing the langsmith documentation, and make sure to set the LANGSMITH_TRACING environment variable to true to enable tracing. You can also set the LANGCHAIN_TRACING_V2 environment variable to true to enable tracing for all LangChain operations.