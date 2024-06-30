# pyenv install 3.10.9 if not installed
# !which pip # /Users/johngavin/.pyenv/shims/pip
# pyenv exec pip install scrapegraphai

# pyenv version # 3.10.9
# !pyenv global 3.10.9
# !pyenv local 3.10.9
# !pyenv which python
# !pyenv exec pip list
# !pyenv exec pip install scrapegraphai
# run helloworld.py using pyenv
# !pyenv exec python helloworld.py

# Playwright for javascript-based scraping:
# playwright install

# download the model on Ollama separately!
# ollama list
# ollama pull mistral


from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {
        "model": "ollama/mistral",
        "temperature": 0,
        "format": "json",  # Ollama needs the format to be specified explicitly
        "base_url": "http://localhost:11434",  # set Ollama URL
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",  # set Ollama URL
    }
}

smart_scraper_graph = SmartScraperGraph(
    prompt="List me all the articles",
    # also accepts a string with the already downloaded HTML code
    source="https://perinim.github.io/projects",
    config=graph_config
)

result = smart_scraper_graph.run()
print(result)
