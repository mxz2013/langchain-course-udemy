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
uv add langchain langchain-openai python-dotenv black isort
```

## 4. Configure Environment Variables

Create a `.env` file at the root of the project and add your API key:

```
OPENAI_API_KEY=your_key_here
```

You can use any other LLM provider's API key as you wish.

## 5. Push Your Branch

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/langchain-course.git
git push --set-upstream origin project/hello-world
```

Replace `YOUR_USERNAME` with your GitHub username.
