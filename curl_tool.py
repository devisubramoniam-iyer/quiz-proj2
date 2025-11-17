import requests
from langchain.tools import tool

@tool
def curl(url: str, method: str = "GET", headers: dict = None, data: dict = None, json: dict = None) -> str:
    """
    A flexible curl-like tool that supports GET, POST, PUT, DELETE.
    The LLM decides method + body.
    """
    try:
        method = method.upper()

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            json=json,
            timeout=10
        )

        # Return first 2000 chars to avoid flooding
        return (
            f"Status: {response.status_code}\n\n"
            f"{response.text[:2000]}"
        )

    except Exception as e:
        return f"Error: {str(e)}"

