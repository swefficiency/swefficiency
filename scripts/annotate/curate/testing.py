import requests

from typing import Optional, List


# Get tags for the images.
def get_docker_hub_tags(
    namespace: str,
    repo: str,
    page_size: int = 100,
    token: Optional[str] = None,
) -> List[str]:
    # Step 1: Get the token
    token_url = "https://ghcr.io/token"
    params = {"scope": f"repository:{namespace}/{repo}:pull"}

    token_response = requests.get(token_url, params=params)
    token_response.raise_for_status()
    token = token_response.json().get("token")

    if not token:
        raise Exception("Failed to retrieve token")

    # Step 2: Use the token to get the list of tags
    page = 1
    all_tags = []

    tags_url = f"https://ghcr.io/v2/{namespace}/{repo}/tags/list?n={page_size}"

    while True:
        headers = {"Authorization": f"Bearer {token}"}

        try:
            tags_response = requests.get(tags_url, headers=headers)
            tags_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("Finished")
            break

        tags_response = requests.get(tags_url, headers=headers)
        tags_response.raise_for_status()
        tags_data = tags_response.json()

        print(page)
        print(tags_data)

        for tag in tags_data.get("tags", []):
            all_tags.append(tag)

        header_next_link = tags_response.headers.get("Link")
        if not header_next_link or 'rel="next"' not in header_next_link:
            break

        url_suffix = header_next_link.split("<", 1)[1].split(">", 1)[0]
        tags_url = "https://ghcr.io" + url_suffix

    return all_tags


print(len(get_docker_hub_tags("TODO", "swefficiency", page_size=100)))
