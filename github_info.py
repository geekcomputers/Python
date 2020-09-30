import requests

_GITHUB_API = "https://api.github.com/user"


def fetch_github_info(auth_user: str, auth_pass: str) -> None:
    # fetching github info using requests
    info = requests.get(_GITHUB_API, auth=(auth_user, auth_pass))

    for g_info, detail in info.json().items():
        print(f"{g_info}: {detail}")


if __name__ == "main":
    fetch_github_info("<USER NAME>", "<PASSWORD>")
