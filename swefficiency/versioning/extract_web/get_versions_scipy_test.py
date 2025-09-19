import requests
import os
from datetime import datetime

# Using the current date for any time-dependent information if needed.
CURRENT_DATE_NOTE = "Script run on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

GITHUB_API_URL = "https://api.github.com"

def get_commit_details_for_tag(repo_owner, repo_name, tag_name, headers):
    """
    Fetches the commit details for a given tag.
    The {ref} in the endpoint /repos/{owner}/{repo}/commits/{ref} can be a branch, tag, or SHA.
    If it's a tag, GitHub resolves it to the commit it points to.
    """
    # Ensure tag_name is properly encoded if it contains special characters,
    # though requests usually handles this. The strip() is good for whitespace.
    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/commits/{tag_name.strip()}"
    # print(f"DEBUG: Fetching commit for tag URL: {url}") # Uncomment for debugging
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    commit_data = response.json()
    return commit_data

def get_releases_and_oldest_commits(repo_owner, repo_name, github_token=None):
    """
    Gets a list of releases from a GitHub repository. For each release,
    it finds the author date of the commit that the release's tag points to.

    The "oldest commit in that release" is interpreted as the commit
    that the release's tag directly points to. The date used is the
    author date of that commit.

    Args:
        repo_owner (str): The owner of the repository (e.g., 'microsoft').
        repo_name (str): The name of the repository (e.g., 'terminal').
        github_token (str, optional): A GitHub Personal Access Token for authentication.
                                      Recommended for private repos or to avoid rate limits.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'release_name': The name of the release.
              - 'tag_name': The tag associated with the release.
              - 'published_at': The publication date of the release (ISO 8601 or "Draft or Not Published").
              - 'commit_author_date': The author date (ISO 8601) of the commit the tag points to.
              - 'commit_sha': The SHA of that commit.
              - 'error': A string message if an error occurred for this specific release/tag.
              If a repository-level error occurs (e.g., repo not found), returns a list with a single error dict.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"  # Good practice to specify API version
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    releases_url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/releases"
    all_releases_info = []  # To store results for each release

    page = 1
    # Max per_page is 100. Using a lower number like 30 can be useful for testing pagination with smaller datasets.
    # For efficiency with many releases, set per_page to 100.
    per_page_count = 100
    print(f"Fetching releases for {repo_owner}/{repo_name} (up to {per_page_count} per page)...")

    while True:
        params = {"page": page, "per_page": per_page_count}
        print(f"Fetching page {page} of releases...")
        try:
            response = requests.get(releases_url, headers=headers, params=params)
            response.raise_for_status()  # Check for errors like 404 (repo not found) or 401/403 (auth issues)
        except requests.exceptions.HTTPError as e:
            error_msg = f"Error fetching releases (page {page}): {e}"
            print(error_msg)
            if page == 1:  # If the first page fails, likely a repo-level or auth issue
                return [{"error": f"Could not fetch releases for {repo_owner}/{repo_name}: {e.response.text if e.response else e}"}]
            break  # Stop if a subsequent page fails, assuming prior pages were processed.
        except requests.exceptions.RequestException as e: # Catch other request exceptions like network errors
            error_msg = f"Network or request error fetching releases (page {page}): {e}"
            print(error_msg)
            if page == 1:
                 return [{"error": f"Could not fetch releases for {repo_owner}/{repo_name}: {e}"}]
            break

        current_page_releases = response.json()
        if not current_page_releases:
            print(f"No more releases found on page {page}.")
            break  # No more releases to fetch

        print(f"Found {len(current_page_releases)} releases on page {page}.")
        for release in current_page_releases:
            release_name = release.get('name', 'N/A')
            tag_name = release.get('tag_name', 'N/A')
            # published_at can be null if the release is a draft
            published_at_raw = release.get('published_at')
            published_at_display = published_at_raw if published_at_raw else "Draft or Not Published"
            
            release_info = {
                "release_name": release_name,
                "tag_name": tag_name,
                "published_at": published_at_display,
                "commit_author_date": None,
                "commit_sha": None
            }

            if tag_name == 'N/A' or not tag_name: # Ensure tag_name is valid
                print(f"Skipping release '{release_name}' due to missing or invalid tag_name.")
                release_info["error"] = "Missing or invalid tag_name for this release."
                all_releases_info.append(release_info)
                continue

            try:
                # print(f"Fetching commit details for release '{release_name}' (tag: {tag_name})...") # Uncomment for verbose logging
                commit_details = get_commit_details_for_tag(repo_owner, repo_name, tag_name, headers)
                
                commit_author_date = commit_details['commit']['author']['date']
                commit_sha = commit_details['sha']
                
                release_info["commit_author_date"] = commit_author_date
                release_info["commit_sha"] = commit_sha
                
            except requests.exceptions.HTTPError as e:
                error_message = f"Error fetching commit for tag '{tag_name}': {e.response.status_code} - {e.response.text if e.response else e}"
                print(error_message)
                release_info["error"] = error_message
            except requests.exceptions.RequestException as e: # Catch other request exceptions
                error_message = f"Network or request error fetching commit for tag '{tag_name}': {e}"
                print(error_message)
                release_info["error"] = error_message
            except KeyError as e:
                error_message = f"Unexpected data structure for commit '{tag_name}': Missing key {e}"
                print(error_message)
                release_info["error"] = error_message
            except Exception as e: # Catch any other unexpected error
                error_message = f"An unexpected error occurred for tag '{tag_name}': {e}"
                print(error_message)
                release_info["error"] = error_message
            
            all_releases_info.append(release_info)

        if len(current_page_releases) < per_page_count:
            # print("Last page of releases reached.") # Uncomment for verbose logging
            break  # Last page reached
        page += 1

    if not all_releases_info and page == 1 and not any(item.get("error") for item in all_releases_info):
        # This handles the case where the repo exists, auth is fine, but there are simply no releases.
        print(f"No releases found for {repo_owner}/{repo_name}.")
        return [] # Return empty list, not an error, if no releases are found.
        
    return all_releases_info

if __name__ == "__main__":
    print(CURRENT_DATE_NOTE)
    repo_owner = input("Enter the repository owner (e.g., 'microsoft'): ").strip()
    repo_name = input("Enter the repository name (e.g., 'terminal'): ").strip()
    
    github_token = os.environ.get("GITHUB_TOKEN")
    
    if not github_token:
        print("\nWarning: GITHUB_TOKEN environment variable not found.")
        print("API calls to GitHub will be unauthenticated.")
        print("This may lead to lower rate limits or inability to access private repositories.")
        if input("Do you want to provide a token now? (yes/no): ").strip().lower() == 'yes':
            github_token = input("Enter your GitHub Personal Access Token: ").strip()

    if not repo_owner or not repo_name:
        print("Repository owner and name cannot be empty. Exiting.")
    else:
        print(f"\nStarting to fetch release information for {repo_owner}/{repo_name}...")
        
        release_commit_data = get_releases_and_oldest_commits(repo_owner, repo_name, github_token)

        print("\n--- Results ---")
        if release_commit_data:
            # Check if the only result is a top-level error (meaning the function itself had an issue early on)
            if len(release_commit_data) == 1 and "error" in release_commit_data[0] and release_commit_data[0].get("release_name") is None:
                print(f"An error occurred: {release_commit_data[0]['error']}")
            else:
                print(f"\nFound information for {len(release_commit_data)} release(s):")
                success_count = 0
                error_count = 0
                for info in release_commit_data:
                    print("-" * 40)
                    print(f"Release Name: {info['release_name']}")
                    print(f"Tag Name:     {info['tag_name']}")
                    print(f"Published At: {info['published_at']}")
                    if info.get("commit_author_date"):
                        print(f"Commit Date:  {info['commit_author_date']} (Author Date)")
                        print(f"Commit SHA:   {info['commit_sha']}")
                        success_count +=1
                    if info.get("error"):
                        print(f"Error:        {info['error']}")
                        error_count +=1
                print("-" * 40)
                print(f"\nSummary: Successfully processed {success_count} releases, {error_count} encountered errors.")
        else:
            # This means get_releases_and_oldest_commits returned an empty list,
            # implying no releases were found (and no repository-level error occurred).
            print(f"No releases were found for {repo_owner}/{repo_name}.")

    print("\nScript finished.")