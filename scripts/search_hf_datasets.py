from huggingface_hub import HfApi

api = HfApi()


def search(query, limit=5):
    print(f"\n=== Search: {query} ===")
    ds = list(api.list_datasets(search=query, limit=limit))
    for d in ds:
        print(f"  • {d.id} (downloads: {getattr(d, 'downloads', 0)})")


search("contract_nli")
search("cuad")
search("lex_glue")
search("injecagent")
search("prompt_injection")
search("sciq")
search("uncertainty")
